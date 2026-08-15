"""Flag reference motions that would themselves trigger the training env's
`done` (fallen) condition — no policy feedback needed.

rgmt terminates an episode when the ROBOT is "fallen" (rgmt/env/track_env.py
::compute_fallen, paper §II-D):

    fallen = pelvis_z < z_fall            (0.12 m)
          OR base_up_z < up_dot_min       (0.0  = tilted past horizontal)
          OR neck_pitch_link_z < head_z_min (0.30 m)

If the REFERENCE itself satisfies that at frame k, then even a perfect tracker
is terminated at k — the clip is unusable as training data no matter how good
the policy gets. This scans the compiled training npz (exactly what training
loads: 23 actuated joints at 30 fps, neck fixed at 0) and reports, per clip,
the fraction of fallen frames and the first fallen frame.

  scan_fallen_refs.py <train_npz_dir> [--out report.json] [--exclude list.json]
                      [--max_frac 0.0] [--workers N]

--exclude writes the clip names to drop (fallen fraction > --max_frac) in the
rejects.json 'DS/sub/clip_poses.mp4' form, ready for compile_training_dataset.
"""
import argparse
import glob
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import multiprocessing as mp

import numpy as np
import mujoco

# thresholds mirror rgmt/configs/env/track.yaml
Z_FALL = 0.12
UP_DOT_MIN = 0.0
HEAD_Z_MIN = 0.30
HEAD_LINK = "neck_pitch_link"
BASE_LINK = "pelvis_link"
# compile_training_dataset keeps these 23 of 25 dofs (neck pitch/yaw dropped),
# so in training the neck joints sit at 0 — mirror that here.
KEPT_DOF = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

_G = {}


def _init():
    import retargeting as g
    m = mujoco.MjModel.from_xml_path(str(g.ROBOT_XML_DICT["asimov"]))
    _G.update(m=m, d=mujoco.MjData(m),
              base=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, BASE_LINK),
              head=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, HEAD_LINK))


def scan_one(npz):
    """Return (clip_name, metrics) for one compiled training clip."""
    if not _G:
        _init()
    m, d, base, head = _G["m"], _G["d"], _G["base"], _G["head"]
    name = os.path.splitext(os.path.basename(npz))[0]
    try:
        z = np.load(npz)
        # rgmt MotionRef npz naming; fall back to the raw retarget-pkl naming
        if "base_frame_pos" in z:
            rp, rr, dp = z["base_frame_pos"], z["base_frame_wxyz"], z["joint_angles"]
        else:
            rp, rr, dp = z["root_pos"], z["root_rot"], z["dof_pos"]   # rr wxyz
        n = len(rp)
        q = np.zeros(m.nq)
        base_z = np.empty(n); up_z = np.empty(n); head_z = np.empty(n)
        for i in range(n):
            q[:3] = rp[i]
            q[3:7] = rr[i]                       # wxyz, as the npz stores it
            q[7:] = 0.0
            q[7 + np.array(KEPT_DOF)] = dp[i]    # neck stays 0, like training
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            base_z[i] = d.xpos[base][2]
            up_z[i] = d.xmat[base].reshape(3, 3)[2, 2]
            head_z[i] = d.xpos[head][2]
        fallen = (base_z < Z_FALL) | (up_z < UP_DOT_MIN) | (head_z < HEAD_Z_MIN)
        first = int(np.argmax(fallen)) if fallen.any() else -1
        return name, {
            "frames": n,
            "fallen_frac": round(float(fallen.mean()), 4),
            "first_fallen": first,
            "min_base_z": round(float(base_z.min()), 3),
            "min_up_z": round(float(up_z.min()), 3),
            "min_head_z": round(float(head_z.min()), 3),
            "cause_base": int((base_z < Z_FALL).sum()),
            "cause_tilt": int((up_z < UP_DOT_MIN).sum()),
            "cause_head": int((head_z < HEAD_Z_MIN).sum()),
        }
    except Exception as e:
        return name, {"error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="compiled training npz dir (motions_train/...)")
    ap.add_argument("--out", default=None, help="write the full per-clip report here")
    ap.add_argument("--exclude", default=None, help="write the drop-list JSON here")
    ap.add_argument("--max_frac", type=float, default=0.0,
                    help="drop clips whose fallen fraction EXCEEDS this (default 0 = any frame)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    npzs = sorted(glob.glob(os.path.join(a.dataset, "*.npz")))
    print(f"{len(npzs)} clips in {a.dataset}", flush=True)
    rep = {}
    with mp.Pool(a.workers, initializer=_init, maxtasksperchild=200) as pool:
        for i, (name, mt) in enumerate(pool.imap_unordered(scan_one, npzs, chunksize=4)):
            rep[name] = mt
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(npzs)}", flush=True)

    ok = {k: v for k, v in rep.items() if "error" not in v}
    bad = [k for k, v in ok.items() if v["fallen_frac"] > a.max_frac]
    errs = [k for k, v in rep.items() if "error" in v]
    any_f = [v for v in ok.values() if v["fallen_frac"] > 0]
    print(f"\nscanned {len(ok)} ({len(errs)} errors)")
    print(f"  clips with ANY fallen frame : {len(any_f)} ({100*len(any_f)/max(len(ok),1):.1f}%)")
    for lo, hi in ((0.0, 0.01), (0.01, 0.05), (0.05, 0.25), (0.25, 1.01)):
        n = sum(1 for v in ok.values() if lo < v["fallen_frac"] <= hi)
        print(f"    fallen fraction {lo:>4.2f}-{hi:<4.2f}: {n}")
    cb = sum(1 for v in ok.values() if v["cause_base"])
    ct = sum(1 for v in ok.values() if v["cause_tilt"])
    ch = sum(1 for v in ok.values() if v["cause_head"])
    print(f"  cause breakdown (clips): pelvis<{Z_FALL} {cb} | tilt past horizontal {ct} | head<{HEAD_Z_MIN} {ch}")
    print(f"  -> {len(bad)} clips exceed --max_frac {a.max_frac}")
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)
        print(f"  report -> {a.out}")
    if a.exclude:
        # clip_name 'DS__sub__clip' -> rejects-style 'DS/sub/clip_poses.mp4'
        drop = []
        for name in sorted(bad):
            parts = name.split("__", 2)
            drop.append("/".join(parts) + "_poses.mp4" if len(parts) == 3 else name)
        json.dump(drop, open(a.exclude, "w"), indent=1)
        print(f"  drop-list ({len(drop)}) -> {a.exclude}")


if __name__ == "__main__":
    main()
