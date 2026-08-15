"""Tag every clip in a retargeted dataset easy/medium/hard, cached to
<dataset>/clip_difficulty.json for the clip explorer and the training handoff.

Five factors per clip:
  joint_vel : largest single-frame joint jump (deg/frame) — manifest `peak_vel`
  root_vel  : max root linear speed (m/s), central difference of `root_pos` so
              a single grounding-step frame can't spike it
  root_ang  : max root angular speed (deg/s), central difference of `root_rot`
              (relative-rotation angle over a 2-frame window)
  sat       : joint-limit saturation % — manifest `overall_sat` (mean % of
              frame x joint cells within ~2 deg of a position limit)
  tilt      : max pelvis tilt from upright (deg), straight from `root_rot`
              (the free-joint quat IS the pelvis orientation; no FK needed)

Worst factor wins: each factor maps to easy(0)/medium(1)/hard(2) via CUTOFFS
(value >= cutoff, inclusive); clip level = max over factors; `driver` names the
factor(s) at that level. Cutoffs are FROZEN — do not recalibrate per release;
re-freeze deliberately with --calibrate and update the provenance note.

  compute_difficulty.py <dataset_dir>               -> <dataset_dir>/clip_difficulty.json
  compute_difficulty.py <dataset_dir> --calibrate   (prints p60/p90 + split; writes nothing)
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from retargeting.utils.clip_names import amass_clip_name

# (medium_at, hard_at) per factor, value >= cutoff (inclusive).
# FROZEN from the reference corpus, ok-status non-glitch cohort (3420 clips),
# p75/p95 of each factor; five-factor split 26% easy / 54% medium / 20% hard.
# Re-frozen after the ankle-roll spec correction (±0.1 -> ±0.35 rad), which
# collapsed the saturation distribution (sat p75 17.4 -> 14.2) and the pelvis
# 0.65 tuning shifted root_vel; joint_vel/root_ang/tilt barely moved. p75/p95
# chosen over p60/p90 because worst-of collapses the easy bucket at p60.
# Re-freeze deliberately (--calibrate + update this note), never silently.
CUTOFFS = {
    "joint_vel": (29.8, 46.6),   # deg/frame at 30 fps (~900 / ~1400 deg/s)
    "root_vel":  (1.32, 2.54),   # m/s (~jog / ~run)
    "root_ang":  (211.3, 466.3), # deg/s (moderate turn / fast spin-whip)
    "sat":       (14.2, 15.9),   # % of frame x joint cells at limit
    "tilt":      (28.7, 48.7),   # deg from upright (deep bend / near-ground)
}
LABELS = ("easy", "medium", "hard")
PCTS = (75, 95)   # calibration percentiles (the frozen CUTOFFS' provenance)


def max_root_speed(root_pos, fps):
    """Max root speed (m/s) via central difference (one-sided at the ends)."""
    p = np.asarray(root_pos, float)
    if len(p) < 2:
        return 0.0
    v = np.empty(len(p))
    v[1:-1] = np.linalg.norm(p[2:] - p[:-2], axis=1) * fps / 2.0
    v[0] = np.linalg.norm(p[1] - p[0]) * fps
    v[-1] = np.linalg.norm(p[-1] - p[-2]) * fps
    return float(v.max())


def max_root_ang_speed(root_rot_xyzw, fps):
    """Max root angular speed (deg/s): relative-rotation angle over a 2-frame
    window (one-sided at the ends), so a single-frame blip can't spike it."""
    q = np.asarray(root_rot_xyzw, float)
    if len(q) < 2:
        return 0.0

    def ang(a, b):   # shortest angle (rad) between orientation pairs, rowwise
        return 2.0 * np.arccos(np.clip(np.abs((a * b).sum(1)), -1.0, 1.0))

    v = np.empty(len(q))
    v[1:-1] = ang(q[:-2], q[2:]) * fps / 2.0
    v[0] = ang(q[:1], q[1:2])[0] * fps
    v[-1] = ang(q[-2:-1], q[-1:])[0] * fps
    return float(np.degrees(v.max()))


def max_tilt_deg(root_rot_xyzw):
    """Max pelvis tilt from upright (deg). root_rot is xyzw (pkl convention)."""
    up_z = Rot.from_quat(np.asarray(root_rot_xyzw, float)).as_matrix()[:, 2, 2]
    return float(np.degrees(np.arccos(np.clip(up_z.min(), -1.0, 1.0))))


def classify(factors, cutoffs=None):
    """(level, label, driver) — worst factor wins; missing factors count easy."""
    cutoffs = CUTOFFS if cutoffs is None else cutoffs
    levels = {}
    for k, (med, hard) in cutoffs.items():
        v = factors.get(k)
        levels[k] = (0 if v is None or med is None
                     else 2 if v >= hard else 1 if v >= med else 0)
    lvl = max(levels.values())
    driver = ",".join(k for k in cutoffs if levels[k] == lvl) if lvl else None
    return lvl, LABELS[lvl], driver


def _clip_name(rel):
    rel = rel[len("_rejected_glitch/"):] if rel.startswith("_rejected_glitch/") else rel
    ds, sub, clip = rel[:-4].split("/", 2)          # strip .pkl
    return f"{ds}__{sub}__{clip[:-6] if clip.endswith('_poses') else clip}"


def _manifest_by_name(dataset):
    """Manifest rows keyed by clip_name (from the AMASS src path)."""
    path = os.path.join(dataset, "manifest.json")
    if not os.path.exists(path):
        return {}
    rows = {}
    for c in json.load(open(path))["clips"]:
        name, _ = amass_clip_name(c["src"])
        rows[name] = c
    return rows


def build(dataset):
    """{clip_name: {factors + status/glitch}} for every pkl in the dataset."""
    manifest = _manifest_by_name(dataset)
    pkls = sorted(glob.glob(os.path.join(dataset, "**", "*.pkl"), recursive=True))
    out = {}
    for i, pkl in enumerate(pkls):
        name = _clip_name(os.path.relpath(pkl, dataset))
        d = pickle.load(open(pkl, "rb"))
        row = manifest.get(name, {})
        out[name] = {
            "joint_vel": row.get("peak_vel"),
            "root_vel": round(max_root_speed(d["root_pos"], float(d["fps"])), 2),
            "root_ang": round(max_root_ang_speed(d["root_rot"], float(d["fps"])), 1),
            "sat": row.get("overall_sat"),
            "tilt": round(max_tilt_deg(d["root_rot"]), 1),
            "_ok": row.get("status") == "ok" and not row.get("glitch", False),
        }
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(pkls)}", flush=True)
    return out


def write_json(dataset, factors):
    data = {}
    for name, f in factors.items():
        lvl, label, driver = classify(f)
        data[name] = {
            "difficulty": label, "level": lvl, "driver": driver,
            "max_joint_vel": f["joint_vel"], "max_root_speed_ms": f["root_vel"],
            "max_root_ang_deg_s": f["root_ang"],
            "sat_pct": f["sat"], "max_tilt_deg": f["tilt"],
        }
    path = os.path.join(dataset, "clip_difficulty.json")
    json.dump(data, open(path, "w"))
    counts = {l: sum(v["difficulty"] == l for v in data.values()) for l in LABELS}
    print(f"DONE -> {path}  ({len(data)} clips: {counts})", flush=True)
    return data


def calibrate(factors):
    """Print p60/p90 per factor over the ok-cohort + the split they'd produce."""
    ok = [f for f in factors.values() if f["_ok"]]
    print(f"cohort: {len(ok)} ok clips of {len(factors)}")
    cuts = {}
    for k in CUTOFFS:
        vals = np.array([f[k] for f in ok if f[k] is not None], float)
        med, hard = (float(np.percentile(vals, p)) for p in PCTS)
        cuts[k] = (round(med, 2), round(hard, 2))
        print(f"  {k:>10}: p{PCTS[0]}={cuts[k][0]:<8} p{PCTS[1]}={cuts[k][1]:<8} "
              f"(min={vals.min():.2f} max={vals.max():.2f})")
    counts = {l: 0 for l in LABELS}
    for f in ok:
        counts[classify(f, cuts)[1]] += 1
    print(f"split over cohort at those cutoffs: "
          f"{ {l: f'{100 * c / len(ok):.0f}%' for l, c in counts.items()} }")
    print(f"CUTOFFS = {json.dumps({k: list(v) for k, v in cuts.items()})}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="retargeted dataset dir (the pkl tree)")
    ap.add_argument("--calibrate", action="store_true",
                    help="print suggested cutoffs from this dataset; write nothing")
    a = ap.parse_args()
    factors = build(a.dataset)
    if a.calibrate:
        calibrate(factors)
    else:
        write_json(a.dataset, factors)


if __name__ == "__main__":
    main()
