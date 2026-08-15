"""Precompute per-clip quality metrics for a retargeted dataset that aren't in
the manifest, cached to <dataset>/clip_metrics.json for the clip explorer.

Per clip:
  float_pct  : % of frames whose lowest robot geom sits > FLOAT_THRESH above the
               floor (robot airborne). From the saved qpos; RSI-grounded clips
               read ~0 except genuine jumps/runs.
  pos_err_cm : mean IK tracking residual -- how far each tracked robot body is
               from the (scaled) human target it was solving for -- computed
               PELVIS-RELATIVE so it is invariant to base placement / grounding.
  foot_err_cm, wrist_err_cm : same, restricted to the foot / wrist bodies.

Parallel, CPU, resumable (skips clips already cached). Run:
  .venv/bin/python clip_explorer/compute_metrics.py <dataset_dir> [--workers N]
"""
import argparse
import glob
import json
import os
import pickle
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # fork-safe; this is CPU work
import multiprocessing as mp

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import smplx_to_asimov as S  # noqa: E402
from smplx_to_asimov import _geom_lowest_z, BODY_MODELS  # noqa: E402

DATASET = CACHE = None   # set in main(); compute_one reads them post-fork

FLOAT_THRESH = 0.02                         # m; lowest geom above this = airborne
FOOT_HUMAN = ("left_foot", "right_foot")
WRIST_HUMAN = ("left_wrist", "right_wrist")

_G = {}   # per-worker singletons (GMR + mujoco model), built lazily


def _init_worker():
    # importing smplx_to_asimov (module S above) already registered the asimov
    # config; nothing to patch here
    import retargeting as g
    m = mujoco.MjModel.from_xml_path(str(g.ROBOT_XML_DICT["asimov"]))
    d = mujoco.MjData(m)
    PLANE = mujoco.mjtGeom.mjGEOM_PLANE
    gids = [gi for gi in range(m.ngeom)
            if m.geom_bodyid[gi] != 0 and m.geom_type[gi] != PLANE]
    corners = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
    _G.update(m=m, d=d, gids=gids, corners=corners)


def _make_gmr(height):
    """A GMR whose human-scale table is baked for this clip's actual height."""
    from retargeting import GeneralMotionRetargeting as GMR
    return GMR(src_human="smplx", tgt_robot="asimov", actual_human_height=height, verbose=False)


def _pelvis_local(pos, origin, R):
    """Express a world position in the frame at `origin` with rotation R (3x3)."""
    return R.T @ (np.asarray(pos) - np.asarray(origin))


def compute_one(pkl):
    """Return (clip_name, metrics dict) for one pkl, or (clip_name, {'error':..})."""
    from scipy.spatial.transform import Rotation as Rot
    from retargeting.utils.smpl import (
        load_smplh_amass_file, get_smplx_data_offline_fast)
    if not _G:
        _init_worker()
    m, d, gids, corners = _G["m"], _G["d"], _G["gids"], _G["corners"]

    # clip identity from the pkl path (handles the _rejected_glitch subtree)
    rel = os.path.relpath(pkl, DATASET)
    rel = rel[len("_rejected_glitch/"):] if rel.startswith("_rejected_glitch/") else rel
    ds, sub, clip = rel[:-4].split("/", 2)          # strip .pkl
    clip_name = f"{ds}__{sub}__{clip[:-6] if clip.endswith('_poses') else clip}"
    src = f"{ROBO}/motions/{ds}/{sub}/{clip}.npz"
    try:
        data = pickle.load(open(pkl, "rb"))
        rp, rr, dp = data["root_pos"], data["root_rot"], data["dof_pos"]
        n = len(rp)
        # --- float %: lowest geom per frame from the saved qpos ---
        airborne = 0
        q = np.empty(m.nq)
        for i in range(n):
            q[:3] = rp[i]; q[3] = rr[i][3]; q[4:7] = rr[i][:3]; q[7:] = dp[i]
            d.qpos[:] = q; mujoco.mj_forward(m, d)
            if min(_geom_lowest_z(m, d, gi, corners) for gi in gids) > FLOAT_THRESH:
                airborne += 1
        float_pct = 100.0 * airborne / max(n, 1)

        # --- pelvis-relative IK tracking residual (GMR scaled for this clip's height) ---
        smplx, bm, out, h = load_smplh_amass_file(src, str(BODY_MODELS))
        frames, _ = get_smplx_data_offline_fast(smplx, bm, out, tgt_fps=30)
        gmr = _make_gmr(h)
        if S.SPINE3_ABOUT_HIPS:            # measure vs the chord-synthesized targets
            S._calibrate_trunk(gmr, frames)
        bid = lambda nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
        hb2bid = {hb: bid(t.frame_name) for hb, t in gmr.human_body_to_task1.items()}
        pelvis_bid = bid(gmr.robot_root_name)
        root = gmr.human_root_name
        errs = {hb: [] for hb in hb2bid}
        for i, fr in enumerate(frames[:n]):
            # robot bodies, pelvis-relative
            q[:3] = rp[i]; q[3] = rr[i][3]; q[4:7] = rr[i][:3]; q[7:] = dp[i]
            d.qpos[:] = q; mujoco.mj_forward(m, d)
            Rr = d.xmat[pelvis_bid].reshape(3, 3); pr = d.xpos[pelvis_bid]
            # human targets, pelvis-relative (with the chord synthesis applied,
            # so residuals are measured against what the IK actually tracked)
            gmr.update_targets(fr)
            if S.SPINE3_ABOUT_HIPS:
                S._spine3_about_hips(gmr, fr)
            tgt = gmr.scaled_human_data
            Rh = Rot.from_quat(tgt[root][1], scalar_first=True).as_matrix(); ph = tgt[root][0]
            for hb, b in hb2bid.items():
                if hb not in tgt:
                    continue
                rl = _pelvis_local(d.xpos[b], pr, Rr)
                hl = _pelvis_local(tgt[hb][0], ph, Rh)
                errs[hb].append(float(np.linalg.norm(rl - hl)))
        def mean_cm(bodies):
            vals = [v for hb in bodies for v in errs.get(hb, [])]
            return round(100.0 * float(np.mean(vals)), 2) if vals else None
        return clip_name, {
            "float_pct": round(float_pct, 1),
            "pos_err_cm": mean_cm(list(errs)),
            "foot_err_cm": mean_cm(FOOT_HUMAN),
            "wrist_err_cm": mean_cm(WRIST_HUMAN),
        }
    except Exception as e:
        return clip_name, {"error": f"{type(e).__name__}: {e}"}


def main():
    global DATASET, CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="retargeted dataset dir (e.g. .../retargeted/tune_v5)")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    DATASET = a.dataset
    CACHE = f"{DATASET}/clip_metrics.json"
    pkls = sorted(glob.glob(os.path.join(DATASET, "**", "*.pkl"), recursive=True))
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = []
    for p in pkls:
        rel = os.path.relpath(p, DATASET)
        rel = rel[len("_rejected_glitch/"):] if rel.startswith("_rejected_glitch/") else rel
        ds, sub, clip = rel[:-4].split("/", 2)
        name = f"{ds}__{sub}__{clip[:-6] if clip.endswith('_poses') else clip}"
        if name not in cache:
            todo.append(p)
    print(f"{len(pkls)} pkls, {len(cache)} cached, {len(todo)} to compute", flush=True)
    done = 0
    with mp.Pool(a.workers, initializer=_init_worker, maxtasksperchild=50) as pool:
        for name, metrics in pool.imap_unordered(compute_one, todo):
            cache[name] = metrics
            done += 1
            if done % 100 == 0 or done == len(todo):
                json.dump(cache, open(CACHE, "w"))     # checkpoint (resumable)
                print(f"  {done}/{len(todo)} (errors so far: "
                      f"{sum('error' in v for v in cache.values())})", flush=True)
    json.dump(cache, open(CACHE, "w"))
    print(f"DONE -> {CACHE}  ({len(cache)} clips)", flush=True)


if __name__ == "__main__":
    main()
