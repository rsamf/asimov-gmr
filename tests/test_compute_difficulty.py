import importlib.util
import json
import pathlib
import pickle

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).parent.parent


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cd = _load("compute_difficulty", "clip_explorer/compute_difficulty.py")


# ---- max pelvis tilt (root_rot is xyzw, scalar-last) ----

def _quat_xyzw(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    h = np.radians(deg) / 2.0
    return np.concatenate([axis * np.sin(h), [np.cos(h)]])


def test_tilt_zero_for_identity_and_pure_yaw():
    rr = np.stack([_quat_xyzw([0, 0, 1], d) for d in (0, 45, 120)])
    assert cd.max_tilt_deg(rr) == pytest.approx(0.0, abs=1e-6)


def test_tilt_matches_roll_and_pitch_angles():
    assert cd.max_tilt_deg(np.stack([_quat_xyzw([1, 0, 0], 90)])) == pytest.approx(90, abs=1e-6)
    assert cd.max_tilt_deg(np.stack([_quat_xyzw([0, 1, 0], 45)])) == pytest.approx(45, abs=1e-6)


def test_tilt_takes_clip_max():
    rr = np.stack([_quat_xyzw([1, 0, 0], d) for d in (5, 60, 20)])
    assert cd.max_tilt_deg(rr) == pytest.approx(60, abs=1e-6)


# ---- max root speed (central difference) ----

def test_root_speed_constant_velocity():
    t = np.arange(60) / 30.0
    p = np.stack([1.5 * t, np.zeros_like(t), np.zeros_like(t)], axis=1)  # 1.5 m/s in x
    assert cd.max_root_speed(p, fps=30.0) == pytest.approx(1.5, rel=1e-6)


def test_root_speed_single_frame_spike_is_halved():
    # one teleport of 0.3 m in an otherwise static clip: plain diff would read
    # 0.3*30 = 9 m/s; the central difference spreads it over 2 frames -> 4.5
    p = np.zeros((20, 3))
    p[10:, 0] = 0.3
    assert cd.max_root_speed(p, fps=30.0) == pytest.approx(4.5, rel=1e-6)


def test_root_speed_degenerate_lengths():
    assert cd.max_root_speed(np.zeros((1, 3)), fps=30.0) == 0.0
    assert cd.max_root_speed(np.zeros((0, 3)), fps=30.0) == 0.0


# ---- max root angular speed (central difference on quats) ----

def test_root_ang_speed_constant_yaw_spin():
    # 3 deg per frame at 30 fps = 90 deg/s, about any axis
    rr = np.stack([_quat_xyzw([0, 0, 1], 3.0 * i) for i in range(20)])
    assert cd.max_root_ang_speed(rr, fps=30.0) == pytest.approx(90.0, rel=1e-6)


def test_root_ang_speed_axis_agnostic():
    rr = np.stack([_quat_xyzw([1, 0, 0], 2.0 * i) for i in range(20)])
    assert cd.max_root_ang_speed(rr, fps=30.0) == pytest.approx(60.0, rel=1e-6)


def test_root_ang_speed_single_frame_blip_is_damped():
    # one 30-deg excursion that returns next frame: plain diff would read
    # 30*30 = 900 deg/s; the 2-frame window sees at most 30 deg over 2 frames -> 450
    rr = np.tile(_quat_xyzw([0, 0, 1], 0), (7, 1))
    rr[3] = _quat_xyzw([0, 1, 0], 30)
    assert cd.max_root_ang_speed(rr, fps=30.0) == pytest.approx(450.0, rel=1e-6)


def test_root_ang_speed_degenerate_lengths():
    assert cd.max_root_ang_speed(np.tile(_quat_xyzw([0, 0, 1], 0), (1, 1)), fps=30.0) == 0.0


# ---- worst-factor labeling ----

TOY = {"joint_vel": (10, 20), "root_vel": (1, 2), "sat": (5, 15), "tilt": (30, 60),
       "root_ang": (120, 240)}


def test_label_easy_when_all_below():
    lvl, label, driver = cd.classify(
        {"joint_vel": 1, "root_vel": 0.1, "sat": 0, "tilt": 3, "root_ang": 20}, TOY)
    assert (lvl, label, driver) == (0, "easy", None)


def test_label_worst_factor_wins_with_driver():
    lvl, label, driver = cd.classify(
        {"joint_vel": 1, "root_vel": 0.1, "sat": 0, "tilt": 75, "root_ang": 20}, TOY)
    assert (lvl, label, driver) == (2, "hard", "tilt")


def test_label_medium_names_all_factors_at_level():
    lvl, label, driver = cd.classify(
        {"joint_vel": 15, "root_vel": 1.5, "sat": 0, "tilt": 3}, TOY)
    assert (lvl, label) == (1, "medium")
    assert driver == "joint_vel,root_vel"


def test_label_cutoff_boundary_is_inclusive():
    lvl, label, _ = cd.classify({"joint_vel": 20, "root_vel": 0, "sat": 0, "tilt": 0}, TOY)
    assert label == "hard"


def test_label_missing_factor_counts_easy():
    lvl, label, driver = cd.classify({"joint_vel": None, "root_vel": 0.1, "sat": 0, "tilt": 3}, TOY)
    assert (lvl, label, driver) == (0, "easy", None)


# ---- end to end on a tmp dataset ----

def _write_pkl(path, root_pos, root_rot, n):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"fps": 30.0, "root_pos": np.asarray(root_pos, float),
                     "root_rot": np.asarray(root_rot, float),
                     "dof_pos": np.zeros((n, 25))}, f)


def test_end_to_end_writes_difficulty_json(tmp_path):
    ds = tmp_path / "ds"
    n = 30
    idq = np.tile(_quat_xyzw([0, 0, 1], 0), (n, 1))
    # calm clip: static, upright, zero manifest factors -> easy under ANY sane cutoffs
    _write_pkl(ds / "CMU/55/calm_poses.pkl", np.zeros((n, 3)), idq, n)
    # wild clip: 90 deg tilt + 8 m/s root -> hard under ANY sane cutoffs
    t = np.arange(n) / 30.0
    fast = np.stack([8.0 * t, np.zeros(n), np.zeros(n)], axis=1)
    tilted = np.tile(_quat_xyzw([1, 0, 0], 90), (n, 1))
    _write_pkl(ds / "_rejected_glitch/CMU/55/wild_poses.pkl", fast, tilted, n)
    manifest = {"summary": {}, "clips": [
        {"src": "/x/motions/CMU/55/calm_poses.npz", "peak_vel": 0.1,
         "overall_sat": 0.0, "status": "ok", "glitch": False},
        {"src": "/x/motions/CMU/55/wild_poses.npz", "peak_vel": 55.0,
         "overall_sat": 40.0, "status": "glitch", "glitch": True},
    ]}
    (ds / "manifest.json").write_text(json.dumps(manifest))

    out = cd.build(str(ds))
    cd.write_json(str(ds), out)

    data = json.loads((ds / "clip_difficulty.json").read_text())
    assert set(data) == {"CMU__55__calm", "CMU__55__wild"}
    calm, wild = data["CMU__55__calm"], data["CMU__55__wild"]
    assert set(calm) == {"difficulty", "level", "driver", "max_joint_vel",
                         "max_root_speed_ms", "sat_pct", "max_tilt_deg",
                         "max_root_ang_deg_s"}
    assert calm["difficulty"] == "easy" and calm["level"] == 0
    assert wild["difficulty"] == "hard" and wild["level"] == 2
    assert wild["max_tilt_deg"] == pytest.approx(90, abs=0.1)
    assert wild["max_root_speed_ms"] == pytest.approx(8.0, rel=0.05)
    assert calm["max_joint_vel"] == 0.1 and wild["sat_pct"] == 40.0
