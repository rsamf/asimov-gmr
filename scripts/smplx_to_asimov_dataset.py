"""Batch-retarget a curated, varied AMASS (SMPL+H) subset onto asimov.

Curation:
  - excludes treadmill clips (feet skate by construction -> bad RL contact data)
  - excludes non-motion files (shape.npz) and hard-to-retarget motions
    (crawl / lying / stairs)
  - balances across datasets (round-robin) for motion variety
Targets ~TARGET_HOURS of source motion; retargets in parallel; writes one .pkl
per clip plus manifest.json. Uses the tuned IK config + partial foot-leveling
in smplx_to_asimov.retarget_clip.
"""
import argparse, json, os, sys, glob, random, time, signal
# Force CPU BEFORE torch is imported (via smplx_to_asimov below). Workers are
# forked, and CUDA cannot be re-initialized in a forked process ("Cannot
# re-initialize CUDA in forked subprocess"); on a GPU box torch.cuda.is_available()
# is True and every forked worker would die. Retargeting is IK/CPU-bound, so CPU
# matches the original run's behavior. spawn is not an alternative: forked workers
# inherit main()'s patched IK_CONFIG_DICT, spawned ones would use the default config.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# --clips_from skips clips the source manifest marked failed (they can hang the
# pool). Set this env var truthy to retry them anyway.
INCLUDE_FAILED_CLIPS = bool(os.environ.get("INCLUDE_FAILED_CLIPS"))
import multiprocessing as mp
import numpy as np
from natsort import natsorted

sys.path.insert(0, os.path.dirname(__file__))
import smplx_to_asimov as _S
from smplx_to_asimov import retarget_clip, FOOT_TILT_ALPHA
import retargeting as _g
import mujoco
from retargeting import ROBOT_XML_DICT
from retargeting.utils.clip_names import amass_rel

CLIP_TIMEOUT = 240  # seconds per clip; a hung clip is skipped (prevents pool stalls).
                    # Raised from 90: the decoupled solve + contact grounding are
                    # slower per clip, and long clips were being spuriously skipped.

def _on_timeout(signum, frame):
    raise TimeoutError("clip exceeded time limit")

# joint limits for per-clip feasibility scoring (built once per worker)
_M = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["asimov"]))
_HINGE = [j for j in range(_M.njnt) if _M.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
_NAMES = [mujoco.mj_id2name(_M, mujoco.mjtObj.mjOBJ_JOINT, j) for j in _HINGE]
_LO = np.array([_M.jnt_range[j][0] for j in _HINGE]); _HI = np.array([_M.jnt_range[j][1] for j in _HINGE])
_ANKLE = [i for i, n in enumerate(_NAMES) if "ankle_roll" in n]
_EPS = 0.035  # ~2 deg


def feasibility(dof_pos):
    atlim = (dof_pos <= _LO + _EPS) | (dof_pos >= _HI - _EPS)
    return float(atlim[:, _ANKLE].any(1).mean() * 100), float(atlim.mean() * 100)


# Glitch detector: an IK solution discontinuity shows up as a per-frame joint
# velocity spike that is an OUTLIER relative to the clip's own motion (peak >>
# p95). Sustained fast motion (running, kicks) has a high p95 too, so its peak
# is not an outlier. Flag if the peak per-frame joint jump is both large in
# absolute terms and an outlier. Returns (peak_deg_per_frame, ratio, is_glitch).
GLITCH_PEAK_DEG = 40.0    # abs floor (deg/frame ~= 1200 deg/s)
GLITCH_RATIO = 3.0        # peak / p95

def glitch_metrics(dof_pos):
    import numpy as _np
    if dof_pos.shape[0] < 5:
        return 0.0, 0.0, False
    v = _np.abs(_np.diff(_np.degrees(dof_pos), axis=0)).max(1)
    pk = float(v.max()); p95 = float(_np.percentile(v, 95))
    ratio = pk / max(p95, 1e-6)
    return pk, ratio, bool(pk > GLITCH_PEAK_DEG and ratio > GLITCH_RATIO)

# AMASS corpus root, laid out <root>/<DATASET>/<subject>/<clip>_poses.npz.
# Obtain it yourself from https://amass.is.tue.mpg.de/ (SMPL+H G variant);
# it is not redistributable. Override with --amass or ASIMOV_AMASS_DIR.
SRC = os.environ.get("ASIMOV_AMASS_DIR", "")
DST = os.environ.get("ASIMOV_OUT_DIR", "")
EXCLUDE_SUBSTR = ["treadmill", "crawl", "_lie", "upstairs", "downstairs", "stagei"]
EXCLUDE_NAMES = {"shape.npz"}
DATASETS = ["ACCAD", "Transitions_mocap", "BMLhandball", "BMLmovi", "BioMotionLab_NTroje",
            "CMU", "Eyes_Japan_Dataset", "MPI_HDM05", "DanceDB"]


def clip_duration(path):
    """Fast source-duration read (no retargeting)."""
    try:
        d = np.load(path, allow_pickle=True)
        return int(d["trans"].shape[0]) / float(d["mocap_framerate"])
    except Exception:
        return None


def candidates_by_dataset():
    out = {}
    for ds in DATASETS:
        files = []
        for p in natsorted(glob.glob(os.path.join(SRC, ds, "**", "*.npz"), recursive=True)):
            name = os.path.basename(p).lower()
            if name in EXCLUDE_NAMES or any(s in name for s in EXCLUDE_SUBSTR):
                continue
            files.append(p)
        out[ds] = files
    return out


def select(target_hours, seed=0):
    """Round-robin across datasets (shuffled) until target source-hours reached."""
    rng = random.Random(seed)
    pools = candidates_by_dataset()
    for ds in pools:
        rng.shuffle(pools[ds])
    selected, total = [], 0.0
    target_s = target_hours * 3600
    order = list(pools.keys())
    idx = {ds: 0 for ds in order}
    progressing = True
    while total < target_s and progressing:
        progressing = False
        for ds in order:
            if idx[ds] < len(pools[ds]):
                p = pools[ds][idx[ds]]; idx[ds] += 1; progressing = True
                dur = clip_duration(p)
                if dur is None or dur < 0.5:   # skip unreadable / trivially short
                    continue
                selected.append((ds, p, dur)); total += dur
                if total >= target_s:
                    break
    return selected, total


def select_from_manifest(manifest_path):
    """Reproduce an existing dataset's EXACT clip set from its manifest.json.

    Used to re-retarget a dataset in place (bypassing select(), whose output has
    drifted with the DATASETS list): every source the manifest recorded, remapped
    off the old /media mount, so downstream artifacts (CSVs, rejects) keep mapping.
    """
    with open(manifest_path) as fh:
        clips = json.load(fh)["clips"]
    selected, total = [], 0.0
    skipped_failed = 0
    for c in clips:
        # Skip clips the source manifest already marked failed. These are
        # pathological (e.g. range-of-motion clips) that time out or HANG in the
        # IK solver; re-attempting them can deadlock the pool (a C-level hang the
        # per-clip SIGALRM timeout can't interrupt). Set INCLUDE_FAILED_CLIPS to retry.
        if not INCLUDE_FAILED_CLIPS and str(c.get("status", "")).startswith("error"):
            skipped_failed += 1
            continue
        # a manifest records absolute paths from the machine that produced it;
        # re-root them onto THIS machine's AMASS corpus
        src = os.path.join(SRC, amass_rel(c["src"]))
        if not os.path.exists(src):
            continue
        ds = amass_rel(src).split("/", 1)[0]
        dur = float(c.get("duration_s") or 0.0)
        selected.append((ds, src, dur))
        total += dur
    if skipped_failed:
        print(f"  (skipped {skipped_failed} clips the source manifest marked failed; "
              f"set INCLUDE_FAILED_CLIPS=1 to retry them)")
    return selected, total


RESUME = False   # set by --resume (fork-inherited by workers)


def _metadata_from_pkl(src, dst, t0):
    """Rebuild a manifest row from an already-retargeted pkl (no IK). Lets
    --resume recover a crashed run: completed clips only need their metadata
    recomputed, which feasibility/glitch derive from the saved dof_pos."""
    import pickle
    with open(dst, "rb") as fh:
        md = pickle.load(fh)
    n = int(md["root_pos"].shape[0])
    fps = float(md["fps"])
    ankle_sat, overall_sat = feasibility(md["dof_pos"])
    peak_v, ratio, is_glitch = glitch_metrics(md["dof_pos"])
    return dict(src=src, dst=dst, frames=n, fps=fps, duration_s=n / fps,
                ankle_roll_sat=round(ankle_sat, 1), overall_sat=round(overall_sat, 1),
                peak_vel=round(peak_v, 1), vel_ratio=round(ratio, 1), glitch=is_glitch,
                status="glitch" if is_glitch else "ok", secs=round(time.time() - t0, 1))


def worker(args):
    src, dst = args
    t0 = time.time()
    if RESUME and os.path.exists(dst) and os.path.getsize(dst) > 0:
        try:
            return _metadata_from_pkl(src, dst, t0)
        except Exception:
            pass   # unreadable/partial pkl -> fall through to a full retarget
    signal.signal(signal.SIGALRM, _on_timeout); signal.alarm(CLIP_TIMEOUT)
    try:
        md, fps = retarget_clip(src, save_pkl=dst)
        signal.alarm(0)
        n = int(md["root_pos"].shape[0])
        ankle_sat, overall_sat = feasibility(md["dof_pos"])
        peak_v, ratio, is_glitch = glitch_metrics(md["dof_pos"])
        return dict(src=src, dst=dst, frames=n, fps=float(fps),
                    duration_s=n / float(fps), ankle_roll_sat=round(ankle_sat, 1),
                    overall_sat=round(overall_sat, 1), peak_vel=round(peak_v, 1),
                    vel_ratio=round(ratio, 1), glitch=is_glitch,
                    status="glitch" if is_glitch else "ok", secs=round(time.time() - t0, 1))
    except Exception as e:
        signal.alarm(0)
        return dict(src=src, dst=dst, frames=0, fps=None, duration_s=0.0,
                    status=f"error: {type(e).__name__}: {e}", secs=round(time.time() - t0, 1))


def main():
    global DST, DATASETS
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_hours", type=float, default=4.5)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--config", default=None,
                    help="IK config: a path, or a filename under ik_configs/. "
                         "Default: the asimov config bundled in NEW/configs "
                         "(registered by importing smplx_to_asimov).")
    ap.add_argument("--leveling", type=int, default=1, help="1=foot-leveling (our config), 0=plain retarget (hybrid)")
    ap.add_argument("--amass", default=None,
                    help="AMASS corpus root (<root>/<DATASET>/<subject>/*_poses.npz); "
                         "defaults to $ASIMOV_AMASS_DIR")
    ap.add_argument("--dst", default=None, help="output dir (default the module DST)")
    ap.add_argument("--datasets", default=None,
                    help="comma-separated dataset subset (default: module DATASETS)")
    ap.add_argument("--clips_from", default=None,
                    help="re-retarget the EXACT clip set of an existing dataset "
                         "(pass its dir or manifest.json), bypassing select()")
    ap.add_argument("--rejects", default=None,
                    help="rejects.json (list of 'DS/sub/clip_poses.mp4' names) whose "
                         "clips are EXCLUDED from the selection")
    ap.add_argument("--clips_list", default=None,
                    help="text file of source .npz paths (one per line) to retarget — "
                         "an explicit selection, e.g. a locomotion-weighted mix")
    ap.add_argument("--resume", action="store_true",
                    help="skip clips whose dst pkl already exists (metadata is "
                         "rebuilt from the pkl) — recover a crashed/killed run")
    a = ap.parse_args()
    global RESUME, SRC
    RESUME = a.resume
    # SRC is read by module-level helpers (candidates_by_dataset), so it must be
    # rebound globally, not shadowed in main()
    if a.amass:
        SRC = a.amass
    if not SRC:
        ap.error("no AMASS corpus: pass --amass <dir> or set ASIMOV_AMASS_DIR.\n"
                 "  Obtain AMASS (SMPL+H G) yourself from https://amass.is.tue.mpg.de/ ;\n"
                 "  expected layout: <dir>/<DATASET>/<subject>/<clip>_poses.npz")
    SRC = os.path.abspath(os.path.expanduser(SRC))
    if not os.path.isdir(SRC):
        ap.error(f"AMASS corpus not found: {SRC}")
    if a.dst: DST = a.dst
    if not DST:
        ap.error("no output dir: pass --dst <dir> or set ASIMOV_OUT_DIR")
    if a.datasets: DATASETS = a.datasets.split(",")
    if a.config:
        import pathlib as _pl
        cfgp = _pl.Path(a.config)
        cfgp = cfgp if cfgp.exists() else _g.IK_CONFIG_ROOT / a.config
        _g.params.IK_CONFIG_DICT["smplx"]["asimov"] = cfgp
        import retargeting.motion_retarget as _mr
        _mr.IK_CONFIG_DICT["smplx"]["asimov"] = cfgp
    _S.LEVEL_FEET_ENABLED = bool(a.leveling)
    print(f"config={_g.params.IK_CONFIG_DICT['smplx']['asimov']} "
          f"leveling={bool(a.leveling)} dst={DST}")

    if a.clips_list:
        selected, src_total, absent = [], 0.0, []
        for line in open(a.clips_list):
            p = line.strip()
            if not p or p.startswith("#"):
                continue
            # entries may be absolute, or AMASS-relative (the portable form
            # curation/clips_*.txt uses, so a list works on any machine)
            if not os.path.isabs(p):
                p = os.path.join(SRC, p)
            if not os.path.exists(p):
                absent.append(p)
                continue
            ds = amass_rel(p).split("/", 1)[0]
            dur = clip_duration(p) or 0.0
            selected.append((ds, p, dur))
            src_total += dur
        if absent:
            print(f"WARNING: {len(absent)} clip(s) in {a.clips_list} are not in "
                  f"{SRC} — your AMASS copy differs from the one this list was "
                  f"built from, e.g. {absent[:2]}")
        print(f"selected {len(selected)} clips from {a.clips_list}")
    elif a.clips_from:
        mpath = a.clips_from
        if os.path.isdir(mpath):
            mpath = os.path.join(mpath, "manifest.json")
        selected, src_total = select_from_manifest(mpath)
        print(f"re-retargeting {len(selected)} clips from {mpath}")
    else:
        selected, src_total = select(a.target_hours, a.seed)
    if a.rejects:
        rej = {os.path.splitext(r)[0] for r in json.load(open(a.rejects))}  # 'DS/sub/clip_poses'
        key = lambda p: os.path.splitext(os.path.relpath(p, SRC))[0]
        before = len(selected)
        selected = [(ds, p, dur) for ds, p, dur in selected if key(p) not in rej]
        src_total = sum(d for _, _, d in selected)
        print(f"  (excluded {before - len(selected)} human-rejected clips via {a.rejects})")
    by_ds = {}
    for ds, p, dur in selected:
        by_ds.setdefault(ds, [0, 0.0]); by_ds[ds][0] += 1; by_ds[ds][1] += dur
    print(f"Selected {len(selected)} clips, {src_total/3600:.2f} h source motion:")
    for ds, (c, s) in by_ds.items():
        print(f"  {ds:24s} {c:5d} clips  {s/3600:5.2f} h")
    if a.dry_run:
        return

    os.makedirs(DST, exist_ok=True)
    tasks = []
    for ds, p, dur in selected:
        dst = p.replace(SRC, DST).replace(".npz", ".pkl")
        tasks.append((p, dst))

    print(f"\nRetargeting with {a.workers} workers ...")
    t0 = time.time()
    results = []
    # maxtasksperchild bounds per-worker RSS growth: without it a worker's memory
    # accumulates over hundreds of clips until the kernel OOM-kills it, and
    # mp.Pool silently respawns the worker while its in-flight task is LOST --
    # imap_unordered then waits forever (observed as a hang at 2400/2589 clips).
    with mp.Pool(a.workers, maxtasksperchild=25) as pool:
        for i, r in enumerate(pool.imap_unordered(worker, tasks)):
            results.append(r)
            if (i + 1) % 25 == 0 or (i + 1) == len(tasks):
                ok = sum(1 for x in results if x["status"] == "ok")
                hrs = sum(x["duration_s"] for x in results if x["status"] == "ok") / 3600
                print(f"  {i+1}/{len(tasks)} done | ok={ok} | {hrs:.2f}h retargeted | {time.time()-t0:.0f}s")

    ok = [r for r in results if r["status"] == "ok"]
    glitchy = [r for r in results if r["status"] == "glitch"]
    # move glitchy clips out of the clean dataset into a rejected/ subtree
    import shutil
    rej_root = os.path.join(DST, "_rejected_glitch")
    for r in glitchy:
        if os.path.exists(r["dst"]):
            rp = os.path.join(rej_root, os.path.relpath(r["dst"], DST))
            os.makedirs(os.path.dirname(rp), exist_ok=True); shutil.move(r["dst"], rp)
    total_h = sum(r["duration_s"] for r in ok) / 3600
    summary = dict(
        clean_clips=len(ok), glitch_clips=len(glitchy), failed=len(results) - len(ok) - len(glitchy),
        total_clips=len(results), clean_hours=round(total_h, 3),
        glitch_hours=round(sum(r["duration_s"] for r in glitchy) / 3600, 3), met_min=total_h >= 3.5,
        per_dataset={ds: round(sum(r["duration_s"] for r in ok
                                   if f"/{ds}/" in r["src"]) / 3600, 3) for ds in DATASETS},
        tuning=dict(foot_tilt_alpha=FOOT_TILT_ALPHA,
                    decouple_arms_from_base=_S.DECOUPLE_ARMS_FROM_BASE,
                    contact_grounding=_S.CONTACT_GROUNDING,
                    ground_clearance_m=_S.GROUND_CLEARANCE,
                    grounding=("contact-mask (PBHC) + %.0fmm clearance"
                               % (_S.GROUND_CLEARANCE * 1000)) if _S.CONTACT_GROUNDING
                              else "exact-geometry global"),
        glitch_filter=dict(peak_deg_per_frame=GLITCH_PEAK_DEG, peak_over_p95=GLITCH_RATIO),
        excluded=EXCLUDE_SUBSTR + sorted(EXCLUDE_NAMES), seconds=round(time.time() - t0, 1),
    )
    with open(os.path.join(DST, "manifest.json"), "w") as fh:
        json.dump({"summary": summary, "clips": sorted(results, key=lambda r: r["src"])}, fh, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
