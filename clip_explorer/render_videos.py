"""Render every clip in a retargeted dataset to a small MP4 (from the saved
qpos — no retargeting). Parallel, resumable. Mirrors the dataset folder layout.

  render_videos.py <dataset_dir> <video_dir> [--clips clips.csv] [--workers N]
"""
import argparse
import glob
import os
import pickle

os.environ.setdefault("MUJOCO_GL", "egl")
import multiprocessing as mp

from retargeting import ROBOT_XML_DICT
from retargeting.utils.clip_names import clip_name_to_rel, read_clip_rows

# resolved lazily: importing this module must not require the robot description
# (the test suite imports it for select_pkls, and a clone may not have asimov-1)
XML = None
W, H = 384, 288

DATASET = OUT = None   # set in main(); render_one reads them post-fork


def select_pkls(dataset, clips_csv=None):
    """Absolute paths of the pkls to render — all of them, or just the CSV's."""
    # exclude _rejected* subtrees RELATIVE to the dataset root, so passing the
    # _rejected_glitch dir itself as <dataset> renders those clips (flattened)
    pkls = sorted(f for f in glob.glob(os.path.join(dataset, "**", "*.pkl"), recursive=True)
                  if "_rejected" not in os.path.relpath(f, dataset))
    if not clips_csv:
        return pkls
    want = {clip_name_to_rel(r["clip_name"]) for r in read_clip_rows(clips_csv)}
    keep = [p for p in pkls if os.path.relpath(p, dataset)[:-4] in want]
    missing = want - {os.path.relpath(p, dataset)[:-4] for p in keep}
    if missing:   # never let a dropped clip look like a rendered one
        print(f"WARNING: {len(missing)} clip(s) in {clips_csv} have no .pkl under "
              f"{dataset}, e.g. {sorted(missing)[:3]}", flush=True)
    return keep


def render_one(pkl):
    import mujoco
    import imageio.v2 as iio
    rel = os.path.relpath(pkl, DATASET)
    out = os.path.join(OUT, rel[:-4] + ".mp4")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return "skip"
    try:
        d = pickle.load(open(pkl, "rb"))
        rp, rr, dp, fps = d["root_pos"], d["root_rot"], d["dof_pos"], float(d["fps"])
        m = mujoco.MjModel.from_xml_path(XML or str(ROBOT_XML_DICT["asimov"]))
        data = mujoco.MjData(m)
        ren = mujoco.Renderer(m, H, W); cam = mujoco.MjvCamera()
        cam.azimuth = 135; cam.elevation = -10; cam.distance = 2.2
        rp = rp.copy(); rp[:, :2] -= rp[0, :2]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        wr = iio.get_writer(out, fps=max(1, round(fps)), macro_block_size=8)
        for i in range(rp.shape[0]):
            data.qpos[:3] = rp[i]
            data.qpos[3] = rr[i][3]; data.qpos[4:7] = rr[i][:3]  # xyzw -> wxyz
            data.qpos[7:] = dp[i]
            mujoco.mj_forward(m, data)
            cam.lookat[:] = [rp[i, 0], rp[i, 1], rp[i, 2] - 0.15]
            ren.update_scene(data, camera=cam); wr.append_data(ren.render())
        wr.close()
        return "ok"
    except Exception as e:
        return f"err {os.path.basename(pkl)}: {e}"


def main():
    global DATASET, OUT, XML
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="retargeted dataset dir (the pkl tree)")
    ap.add_argument("out", help="video output dir (mirrors the dataset layout)")
    ap.add_argument("--clips", help="CSV with a clip_name column; render only those clips")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    # render_one reads these as globals; mp forks on Linux, so children inherit them
    DATASET, OUT = a.dataset, a.out
    XML = str(ROBOT_XML_DICT["asimov"])   # fail fast, before forking workers
    pkls = select_pkls(DATASET, a.clips)
    print(f"{len(pkls)} clips -> {OUT}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    done = ok = skip = err = 0
    with mp.Pool(a.workers) as pool:
        for r in pool.imap_unordered(render_one, pkls):
            done += 1
            ok += r == "ok"; skip += r == "skip"; err += r.startswith("err")
            if r.startswith("err"):
                print(f"  {r}", flush=True)
            if done % 100 == 0:
                print(f"  {done}/{len(pkls)} (ok={ok} skip={skip} err={err})", flush=True)
    print(f"DONE rendered ok={ok} skip={skip} err={err}", flush=True)


if __name__ == "__main__":
    main()
