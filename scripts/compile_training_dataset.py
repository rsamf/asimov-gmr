"""Compile the cleaned asimov retarget into the rgmt MotionRef training format.

Per clip -> one .npz with exactly three arrays (shared first dim F):
  base_frame_pos  (F,3)  root world position, meters, (x,y,z)   [already grounded]
  base_frame_wxyz (F,4)  root quaternion, wxyz (w first)
  joint_angles    (F,23) actuated joints, radians, ASIMOV_ACTUATED_JOINT_NAMES order

Source: the hybrid retarget pkls (root_pos, root_rot[xyzw], dof_pos[25], fps).
Steps: drop rejected clips, resample to exactly 30 fps (so src_fps | physics_fps),
select the 23 actuated joints (drop the 2 neck joints), xyzw->wxyz, drop sub-min clips.
"""
import argparse, glob, json, os, pickle
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R, Slerp
from retargeting import ROBOT_XML_DICT

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# canonical 23 actuated joints (rgmt ASIMOV_ACTUATED_JOINT_NAMES), right-arm before left
TARGET = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_yaw",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_yaw",
]


def joint_columns():
    """Map each TARGET joint to its column in dof_pos (= qpos[7:]) order."""
    m = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["asimov"]))
    name2col = {}
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).replace("_joint", "")
            name2col[nm] = int(m.jnt_qposadr[j] - 7)
    missing = [n for n in TARGET if n not in name2col]
    assert not missing, f"target joints not in model: {missing}"
    cols = [name2col[n] for n in TARGET]
    assert len(cols) == 23, len(cols)
    assert "neck_yaw" not in TARGET and "neck_pitch" not in TARGET
    return cols


def resample(rp, rr_xyzw, dp, fps, tgt_fps):
    """Resample arrays from `fps` to exactly `tgt_fps`. Linear pos/joints, SLERP quat."""
    n = rp.shape[0]
    if n < 2:
        return None
    if abs(fps - tgt_fps) < 1e-6:
        return rp, rr_xyzw, dp
    dur = (n - 1) / fps
    m = int(np.floor(dur * tgt_fps)) + 1
    t_old = np.arange(n) / fps
    t_new = np.arange(m) / tgt_fps
    rp2 = np.column_stack([np.interp(t_new, t_old, rp[:, k]) for k in range(3)])
    dp2 = np.column_stack([np.interp(t_new, t_old, dp[:, k]) for k in range(dp.shape[1])])
    rr2 = Slerp(t_old, R.from_quat(rr_xyzw))(t_new).as_quat()  # xyzw in / out
    return rp2, rr2, dp2


def load_difficulty(dataset):
    """{clip_name: label} from <dataset>/clip_difficulty.json (the clip
    explorer's difficulty builder), {} if the file doesn't exist."""
    p = os.path.join(dataset, "clip_difficulty.json")
    if not os.path.exists(p):
        return {}
    return {k: v["difficulty"] for k, v in json.load(open(p)).items()}


def load_test_split(path):
    """Set of canonical held-out TEST stems from tune/test_split.json
    (make_test_split.py), or None if the file doesn't exist."""
    if not path or not os.path.exists(path):
        return None
    return {s for lst in json.load(open(path))["test"].values() for s in lst}


def apply_split(index, test_set, out, canonical_path):
    """Stamp split=train/test on index entries and write <out>/split.json in
    the training repo's schema ({"train": [stems], "test": [stems]} — its
    loader raises on names absent from the corpus, so `test` holds only clips
    THIS release kept; canonical test clips it dropped are reported as holes."""
    kept = {e["clip"] for e in index}
    for e in index:
        e["split"] = "test" if e["clip"] in test_set else "train"
    missing = sorted(test_set - kept)
    sj = {"train": sorted(kept - test_set), "test": sorted(kept & test_set),
          "n_train": len(kept - test_set), "n_test": len(kept & test_set),
          "source": canonical_path, "missing_test_clips": missing}
    json.dump(sj, open(os.path.join(out, "split.json"), "w"), indent=2)
    if missing:
        print(f"WARNING: {len(missing)} canonical test clip(s) not in this "
              f"release (holes, not topped up): {missing[:5]}")
    print(f"split.json: {sj['n_train']} train / {sj['n_test']} test -> {out}")


def annotate_summary(dataset, out, test_split_path):
    """Join difficulty + split into an EXISTING <out>/compile_summary.json in
    place — touches no npz. For releases whose exact compile inputs are no
    longer reproducible (the rejects list moves on after a release)."""
    diff = load_difficulty(dataset)
    path = os.path.join(out, "compile_summary.json")
    summary = json.load(open(path))
    hit = 0
    for e in summary["index"]:
        if e["clip"] in diff:
            e["difficulty"] = diff[e["clip"]]
            hit += 1
    test_set = load_test_split(test_split_path)
    if test_set is not None:
        apply_split(summary["index"], test_set, out, test_split_path)
    json.dump(summary, open(path, "w"), indent=2)
    print(f"annotated {hit}/{len(summary['index'])} index entries -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="retargeted dataset dir (the pkl tree + manifest.json)")
    ap.add_argument("--rejects", default=os.path.join(REPO, "curation", "rejects.json"),
                    help="clip names to exclude")
    ap.add_argument("--out", required=True, help="training-set output dir")
    ap.add_argument("--tgt_fps", type=float, default=30.0)
    ap.add_argument("--annotate_only", action="store_true",
                    help="only (re)join difficulty + split into the existing "
                         "compile_summary.json in --out; write no npz")
    ap.add_argument("--test_split",
                    default=os.path.join(REPO, "curation", "test_split.json"),
                    help="canonical held-out test list (make_test_split.py); "
                         "stamps split=train/test and emits <out>/split.json")
    # No arbitrary duration floor: the only real constraint is the training env's,
    # derived below (MIN_SOURCE_FRAMES). A seconds-based floor kept discarding
    # short-but-valid clips -- at 1.0 s it silently dropped 13 KIT running clips
    # (0.73-0.99 s), and AMASS running comes in 1-4 s bursts, the scarcest data in
    # the corpus. Set --min_seconds only to impose an EXTRA restriction.
    ap.add_argument("--min_seconds", type=float, default=0.0)
    a = ap.parse_args()

    if a.annotate_only:
        annotate_summary(a.dataset, a.out, a.test_split)
        return

    diff = load_difficulty(a.dataset)
    cols = joint_columns()
    assert cols == [0,1,2,3,4,5,6,7,8,9,10,11,12,15,16,17,18,19,20,21,22,23,24], cols
    rej = set(json.load(open(a.rejects))) if os.path.exists(a.rejects) else set()
    # Hard floor from the training env, not from taste: sampling needs
    # room >= _max_lookahead = L+1 = 11 UPSAMPLED frames (rgmt track_env.py), and
    # refs upsample 30->60 Hz as T_up = (T_src-1)*UPSAMPLE + 1. Solving
    # T_up - 1 >= MAX_LOOKAHEAD gives T_src >= 7: below that a clip yields ZERO
    # valid start frames and is genuinely unusable (it would also make the
    # finite-difference velocities degenerate).
    UPSAMPLE, MAX_LOOKAHEAD = 2, 11
    MIN_SOURCE_FRAMES = -(-MAX_LOOKAHEAD // UPSAMPLE) + 1          # ceil-div -> 7
    min_frames = max(MIN_SOURCE_FRAMES, int(round(a.min_seconds * a.tgt_fps)))
    os.makedirs(a.out, exist_ok=True)

    pkls = [p for p in glob.glob(os.path.join(a.dataset, "**", "*.pkl"), recursive=True)
            if "_rejected" not in p]
    written = skip_rej = skip_short = err = 0
    total_frames = 0
    durations = []
    index = []
    for p in sorted(pkls):
        rel = os.path.relpath(p, a.dataset)
        if rel[:-4] + ".mp4" in rej:
            skip_rej += 1
            continue
        try:
            d = pickle.load(open(p, "rb"))
            out = resample(d["root_pos"].astype(np.float64), d["root_rot"].astype(np.float64),
                           d["dof_pos"].astype(np.float64), float(d["fps"]), a.tgt_fps)
            if out is None:
                skip_short += 1
                continue
            rp, rr, dp = out
            F = rp.shape[0]
            if F < min_frames:
                skip_short += 1
                continue
            wxyz = rr[:, [3, 0, 1, 2]]
            wxyz = wxyz / np.linalg.norm(wxyz, axis=1, keepdims=True)
            ja = dp[:, cols]
            assert ja.shape == (F, 23)
            assert rp.shape == (F, 3) and wxyz.shape == (F, 4)
            stem = rel[:-4].replace(os.sep, "__")
            if stem.endswith("_poses"):
                stem = stem[:-6]
            np.savez(os.path.join(a.out, stem + ".npz"),
                     base_frame_pos=rp.astype(np.float32),
                     base_frame_wxyz=wxyz.astype(np.float32),
                     joint_angles=ja.astype(np.float32))
            written += 1
            total_frames += F
            durations.append(F / a.tgt_fps)
            entry = {"clip": stem, "src": rel, "frames": F, "seconds": round(F / a.tgt_fps, 2)}
            if stem in diff:
                entry["difficulty"] = diff[stem]
            index.append(entry)
        except Exception as e:
            err += 1
            print("ERR", rel, repr(e))

    test_set = load_test_split(a.test_split)
    if test_set is not None:
        apply_split(index, test_set, a.out, a.test_split)
    dur = np.array(durations)
    json.dump({"clips": written, "frames": int(total_frames),
               "hours": round(float(dur.sum()) / 3600, 3), "fps": a.tgt_fps,
               "min_seconds": a.min_seconds, "skipped_rejected": skip_rej,
               "skipped_short": skip_short, "errors": err, "index": index},
              open(os.path.join(a.out, "compile_summary.json"), "w"), indent=2)
    print(f"\nWROTE {written} clips -> {a.out}")
    print(f"  frames={total_frames}  hours={dur.sum()/3600:.2f}  fps={a.tgt_fps:g}")
    print(f"  skipped: rejected={skip_rej}  unusable(<{min_frames} frames)={skip_short}  errors={err}")
    if len(dur):
        print(f"  clip seconds: min={dur.min():.1f} median={np.median(dur):.1f} max={dur.max():.1f}")


if __name__ == "__main__":
    main()
