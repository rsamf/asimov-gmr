"""Per-clip joint-feasibility scan over a retargeted asimov dataset.

For each .pkl, computes the fraction of frames where each joint sits within
~2 deg of a limit, focusing on ankle-roll (asimov's tightest DOF). Reports the
distribution and how many feasible hours survive at various ankle-roll-saturation
thresholds, so curation can keep only physically-natural-for-asimov clips.
"""
import argparse, glob, os, pickle, json
import numpy as np
import mujoco
from retargeting import ROBOT_XML_DICT

EPS = 0.035  # ~2 deg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="retargeted dataset dir to scan")
    ap.add_argument("--out", default=None, help="write per-clip feasibility json")
    a = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["asimov"]))
    hinges = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinges]
    lo = np.array([m.jnt_range[j][0] for j in hinges]); hi = np.array([m.jnt_range[j][1] for j in hinges])
    ar = [i for i, n in enumerate(names) if "ankle_roll" in n]

    fs = sorted(glob.glob(os.path.join(a.dir, "**", "*.pkl"), recursive=True))
    rows = []
    for f in fs:
        d = pickle.load(open(f, "rb")); dp = d["dof_pos"]; n = dp.shape[0]
        atlim = (dp <= lo + EPS) | (dp >= hi - EPS)            # (n,25) bool
        ankle = atlim[:, ar].any(1).mean() * 100               # % frames any ankle-roll at limit
        overall = atlim.mean() * 100                           # mean joints-at-limit fraction
        rows.append(dict(f=f, frames=n, dur=n / d["fps"], ankle_roll_sat=ankle, overall_sat=overall))
    tot_h = sum(r["dur"] for r in rows) / 3600
    ank = np.array([r["ankle_roll_sat"] for r in rows])
    print(f"clips={len(rows)}  total={tot_h:.2f}h")
    print(f"ankle-roll saturation per clip: median={np.median(ank):.0f}%  p25={np.percentile(ank,25):.0f}%  p75={np.percentile(ank,75):.0f}%  p90={np.percentile(ank,90):.0f}%")
    print("feasible hours retained at ankle-roll-saturation thresholds:")
    for th in [5, 10, 15, 20, 30, 50, 100]:
        keep = [r for r in rows if r["ankle_roll_sat"] <= th]
        print(f"  <= {th:3d}% : {len(keep):5d} clips  {sum(r['dur'] for r in keep)/3600:5.2f} h")
    if a.out:
        json.dump(rows, open(a.out, "w"))
        print("wrote", a.out)

if __name__ == "__main__":
    main()
