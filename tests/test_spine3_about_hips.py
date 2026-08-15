"""Unit tests for the spine3-about-hips target synthesis (pure math, no models).

"spine3 rotates about the hips only" = two identities on the synthesized targets:
  I1 tilt : the torso target z-axis equals the chord hip_center->spine3
  I2 pivot: the base position target moves rigidly about the hip center (the
            same minimal rotation dR that retilts the torso also swings the
            base point about hip_c)
"""
import pathlib
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import smplx_to_asimov as S  # noqa: E402


def test_rot_between_maps_a_to_b():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.normal(size=3); a /= np.linalg.norm(a)
        b = rng.normal(size=3); b /= np.linalg.norm(b)
        assert np.allclose(S._rot_between(a, b).apply(a), b, atol=1e-9)


def test_rot_between_is_minimal():
    a = np.array([0.0, 0.0, 1.0])
    b = np.array([np.sin(0.3), 0.0, np.cos(0.3)])
    R = S._rot_between(a, b)
    assert R.magnitude() == pytest.approx(0.3, abs=1e-9), "must not add twist"


def test_rot_between_parallel_and_antiparallel():
    z = np.array([0.0, 0.0, 1.0])
    assert S._rot_between(z, z).magnitude() == pytest.approx(0.0)
    R = S._rot_between(z, -z)
    assert np.allclose(R.apply(z), -z, atol=1e-9)


class _FakeTask:
    def set_target(self, t):
        self.target = t


class _FakeGMR:
    """Just enough of GMR for _spine3_about_hips: scaled data + task tables.
    scale_human_data is identity, so clean anatomy == task space in these tests
    (the offset-exclusion path is exercised by the pos_offset test below)."""
    human_root_name = "pelvis"
    human_scale_table = {}

    def __init__(self, sd):
        self.scaled_human_data = sd
        self.human_body_to_task1 = {"pelvis": _FakeTask(), "spine3": _FakeTask()}
        self.human_body_to_task2 = {}

    def scale_human_data(self, frame, root, table):
        return frame


def _mk(sd, frame=None):
    g = _FakeGMR(sd)
    S._spine3_about_hips(g, frame if frame is not None else sd)
    return g


def _sd(hip_c, chord_dir, chord_len, s3_quat, pv_offset, pv_quat):
    """Build scaled_human_data with spine3 at hip_c + chord, pelvis at hip_c + pv_offset."""
    hip_c = np.asarray(hip_c, float)
    s3 = hip_c + chord_len * np.asarray(chord_dir, float)
    return {
        "left_hip": (hip_c + [0.0, 0.08, 0.0], np.array([1.0, 0, 0, 0])),
        "right_hip": (hip_c - [0.0, 0.08, 0.0], np.array([1.0, 0, 0, 0])),
        "spine3": (s3, np.asarray(s3_quat, float)),
        "pelvis": (hip_c + np.asarray(pv_offset, float), np.asarray(pv_quat, float)),
    }


def test_backleaned_frame_with_vertical_chord_uprights_the_torso():
    # spine3 point directly above the hips (chord vertical) but its FRAME tipped
    # 17 deg back (sacral slope): no rotation about the hips has occurred, so the
    # synthesized torso target must be upright and the base target unmoved.
    tip = Rot.from_euler("y", -17, degrees=True).as_quat(scalar_first=True)
    sd = _sd([1.0, 2.0, 0.9], [0, 0, 1.0], 0.4, tip, [0.0, 0.0, 0.09], [1.0, 0, 0, 0])
    pv0 = sd["pelvis"][0].copy()
    g = _mk(sd)
    q = g.scaled_human_data["spine3"][1]
    z = Rot.from_quat(q, scalar_first=True).apply([0.0, 0.0, 1.0])
    assert np.allclose(z, [0, 0, 1], atol=1e-6), "torso target must be upright"
    assert np.allclose(g.scaled_human_data["pelvis"][0], pv0, atol=1e-6), \
        "vertical chord + frame-only backlean must not move the base target"


def test_tilted_chord_sets_torso_z_and_swings_base_about_hip_center():
    # chord pitched 60 deg forward, spine3 frame pitched 75 (lumbar flex on top):
    # torso z must equal the CHORD (not the frame), and the base target must be
    # the pelvis point rotated about hip_c by dR = rot_between(frame_z, chord).
    hip_c = np.array([0.5, -0.3, 0.85])
    chord = np.array([np.sin(np.radians(60)), 0.0, np.cos(np.radians(60))])
    fq = Rot.from_euler("y", 75, degrees=True).as_quat(scalar_first=True)
    pv_off = np.array([0.02, 0.0, 0.09])
    sd = _sd(hip_c, chord, 0.38, fq, pv_off, [1.0, 0, 0, 0])
    g = _mk(sd)
    zq = Rot.from_quat(g.scaled_human_data["spine3"][1], scalar_first=True)
    assert np.allclose(zq.apply([0, 0, 1.0]), chord, atol=1e-6), "I1: torso z == chord"
    expect = hip_c + np.linalg.norm(pv_off) * chord
    assert np.allclose(g.scaled_human_data["pelvis"][0], expect, atol=1e-6), \
        "I2: base target rides the chord at its rigid arm length above hip_c"
    # rigid-arm invariant: distance from base target to hip center is preserved
    assert np.linalg.norm(g.scaled_human_data["pelvis"][0] - hip_c) == \
        pytest.approx(np.linalg.norm(pv_off), abs=1e-9)


def test_task_offsets_do_not_tilt_the_chord():
    # Task-space hips pushed 5 cm FORWARD (link-placement pos_offsets, the
    # robot-leans-back bug): the chord direction must come from the CLEAN
    # frame, so the synthesized torso target stays vertical; only the
    # PLACEMENT anchors at the task-space hip center.
    hip_c = np.array([0.0, 0.0, 0.9])
    clean = _sd(hip_c, [0, 0, 1.0], 0.40, [1.0, 0, 0, 0], [0.0, 0.0, 0.09], [1.0, 0, 0, 0])
    task = {k: (np.asarray(p).copy(), q.copy()) for k, (p, q) in clean.items()}
    for k in ("left_hip", "right_hip"):
        task[k] = (task[k][0] + [0.05, 0.0, 0.0], task[k][1])   # offset task hips
    g = _FakeGMR(task)
    S._spine3_about_hips(g, clean)
    zq = Rot.from_quat(g.scaled_human_data["spine3"][1], scalar_first=True)
    assert np.allclose(zq.apply([0, 0, 1.0]), [0, 0, 1.0], atol=1e-9), \
        "torso target must stay vertical despite offset task hips"
    task_hip_c = hip_c + [0.05, 0.0, 0.0]
    assert np.allclose(g.scaled_human_data["pelvis"][0], task_hip_c + [0, 0, 0.09], atol=1e-9), \
        "base target must anchor at the TASK-space hip center"


def test_calibrated_radii_defeat_scrunch():
    # the human trunk chord scrunches (0.28 here vs the calibrated upright 0.40)
    # and the offset-corrupted pelvis arm reads 0.065 vs the calibrated 0.136 --
    # with _trunk_calib set, both targets must sit at the RIGID radii on the
    # live chord direction, so the drawn trunk length never changes.
    hip_c = np.array([0.2, 0.1, 0.8])
    chord = np.array([np.sin(np.radians(70)), 0.0, np.cos(np.radians(70))])
    sd = _sd(hip_c, chord, 0.28, [1.0, 0, 0, 0], [0.02, 0.0, 0.06], [1.0, 0, 0, 0])
    g = _FakeGMR(sd)
    g._trunk_calib = (0.136, 0.40)
    S._spine3_about_hips(g, sd)
    pv = g.scaled_human_data["pelvis"][0]
    s3 = g.scaled_human_data["spine3"][0]
    assert np.linalg.norm(pv - hip_c) == pytest.approx(0.136, abs=1e-9)
    assert np.linalg.norm(s3 - hip_c) == pytest.approx(0.40, abs=1e-9)
    assert np.linalg.norm(s3 - pv) == pytest.approx(0.40 - 0.136, abs=1e-9), \
        "trunk length must be rigid regardless of source scrunch"
    # both on the live chord direction
    assert np.allclose((s3 - hip_c) / 0.40, chord, atol=1e-9)


def test_degenerate_chord_keeps_raw_targets():
    sd = _sd([0, 0, 1.0], [0, 0, 1.0], 0.01, [1.0, 0, 0, 0],   # 1 cm chord
             [0, 0, 0.09], [1.0, 0, 0, 0])
    pv0, s30 = sd["pelvis"][0].copy(), sd["spine3"][0].copy()
    g = _mk(sd)
    assert not hasattr(g.human_body_to_task1["pelvis"], "target"), \
        "sub-MIN_CHORD frames must not be resynthesized"
    assert np.allclose(g.scaled_human_data["pelvis"][0], pv0)
    assert np.allclose(g.scaled_human_data["spine3"][0], s30)
