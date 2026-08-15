"""Hand-tune an IK config against KIT/12: retarget with your config and play the
result back in an INTERACTIVE MuJoCo viewer (kinematic — no physics, just poses).

Workflow:
  1. edit  NEW/configs/smplx_to_asimov_tune.json
  2. run   python scripts/retarget_and_view.py                     # all 48 KIT/12 clips
     or    python scripts/retarget_and_view.py --clip LeftTurn03   # iterate on one
     or    python scripts/retarget_and_view.py --clip LeftTurn03 --overlay   # + SMPL-H skeleton
  3. orbit/zoom in the window; close it (or Ctrl-C), tweak the config, re-run.

--overlay draws the (scaled) SMPL-H joint skeleton in orange on top of the robot, so you
can see whether each human target lands on the corresponding robot link — the direct
signal for tuning human_scale_table.

Flags:
  --config <name|path>   IK config (default: NEW/configs/smplx_to_asimov_tune.json)
  --leveling 0|1         full asimov pipeline: foot-leveling + decouple + base-init +
                         foot-pinning + yaw-only pelvis (default 1, matches the batch;
                         0 = plain GMR, for comparison only)
  --clip <substr>        only clips whose filename contains this
  --overlay              draw the scaled SMPL-H skeleton overlay
  --speed <x>            playback speed multiplier (default 1.0)
  --no_view              retarget only, print stats, don't open the viewer
  --render <mp4|dir>     headless: write an MP4 per clip via EGL (no display) — for SSH.
                         Add --render_orbit <deg/s> to spin the camera. scp/serve the file.

The interactive viewer needs a display (GLFW window) — run it in your own session, or
over SSH use --render (works headless on the GPU) and view the MP4. See tuning notes.
"""
import argparse, glob, os, sys, time, pathlib
# --render is headless (offscreen EGL); the interactive viewer needs GLFW+display.
# MUJOCO_GL must be chosen before mujoco is imported, so key off argv here.
RENDER_MODE = "--render" in sys.argv
if RENDER_MODE:
    os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
if not RENDER_MODE:
    import mujoco.viewer

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import retargeting as g
from retargeting import GeneralMotionRetargeting as GMR
from retargeting import motion_retarget as mr
from retargeting.utils.smpl import load_smplh_amass_file, get_smplx_data_offline_fast
import smplx_to_asimov as S

BODY_MODELS = str(HERE.parent / "assets" / "body_models")
MOTIONS = os.environ.get("ASIMOV_AMASS_DIR", "")
# clip source dir (override with --dir). KIT/12 = the turn/walk set we tune on.
SRC_DIR = f"{MOTIONS}/KIT/12"

# skeleton connectivity among the scale-table joints (what human_scale_table controls)
BONES = [
    ("pelvis", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_foot"),
    ("pelvis", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_foot"),
    ("pelvis", "spine3"),
    ("spine3", "left_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("spine3", "right_shoulder"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
]
_ORANGE = (np.array([1, 0.55, 0.1, 0.95], np.float32), np.array([1, 0.55, 0.1, 0.5], np.float32))
_MAGENTA = (np.array([1, 0.2, 0.8, 0.95], np.float32), np.array([1, 0.2, 0.8, 0.5], np.float32))
_EYE = np.eye(3).flatten()
ROBOT_LINK_MAP = {  # scale-table joints have actual IK targets; used to pull target positions
    "pelvis", "spine3", "left_hip", "left_knee", "left_foot", "right_hip", "right_knee", "right_foot",
    "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist",
}


def draw_overlay(scn, joints, colors, reset=False, r=0.025):
    """Append SMPL-H joints (spheres) + bones (capsules) to a MjvScene.

    `scn` is the live viewer's `user_scn` (interactive) or the offscreen
    `Renderer.scene` (--render); the caller passes whichever applies.
    """
    if reset:
        scn.ngeom = 0
    jcol, bcol = colors
    n = scn.ngeom
    for p in joints.values():
        if n >= scn.maxgeom:
            break
        mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([r, 0, 0]), p.astype(np.float64), _EYE, jcol)
        n += 1
    for a, b in BONES:
        if a in joints and b in joints and n < scn.maxgeom:
            mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_CAPSULE,
                                np.zeros(3), np.zeros(3), _EYE, bcol)
            mujoco.mjv_connector(scn.geoms[n], mujoco.mjtGeom.mjGEOM_CAPSULE, 0.009,
                                 joints[a].astype(np.float64), joints[b].astype(np.float64))
            n += 1
    scn.ngeom = n


def _style_collision(model, mesh_alpha):
    """Make the visual mesh transparent and paint collision primitives green."""
    for gi in range(model.ngeom):
        if model.geom_bodyid[gi] == 0 or model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        model.geom_matid[gi] = -1
        if model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH:
            model.geom_rgba[gi] = [0.7, 0.7, 0.75, mesh_alpha]
        else:
            model.geom_rgba[gi] = [0.2, 0.95, 0.4, 0.95]


def render_seqs(seqs, xml, a):
    """Headless offscreen render of each retargeted clip to an MP4 (EGL, no display).

    `a.render` is an .mp4 path (single clip) or a directory (one .mp4 per clip).
    Camera tracks the robot in xy at a fixed lookat height so grounding/float is
    visible; --render_orbit spins the azimuth for a 3D reveal. --overlay draws the
    same orange/magenta SMPL-H skeleton the interactive viewer shows.
    """
    import imageio.v2 as iio
    model = mujoco.MjModel.from_xml_path(xml)
    _style_collision(model, a.mesh_alpha)
    data = mujoco.MjData(model)
    ren = mujoco.Renderer(model, a.render_h, a.render_w)
    cam = mujoco.MjvCamera()
    cam.distance = g.VIEWER_CAM_DISTANCE_DICT.get("asimov", 2.5)
    cam.elevation = a.render_elev
    opt = mujoco.MjvOption()
    if a.collision:
        opt.geomgroup[3] = 1
    single = a.render.endswith(".mp4") and len(seqs) == 1
    outdir = os.path.dirname(a.render) if single else a.render
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    for name, q, fps, skel, tgt in seqs:
        path = a.render if single else os.path.join(a.render, name + ".mp4")
        wr = iio.get_writer(path, fps=max(1, round(fps * a.speed)), macro_block_size=1)
        for i in range(len(q)):
            data.qpos[:] = q[i]
            mujoco.mj_forward(model, data)
            cam.azimuth = a.render_az + (i / fps) * a.render_orbit
            cam.lookat[:] = [q[i, 0], q[i, 1], a.render_z]
            ren.update_scene(data, camera=cam, scene_option=opt)
            if a.overlay and skel:
                draw_overlay(ren.scene, skel[i], _ORANGE, r=0.025)   # scaled human (pre-offset)
                draw_overlay(ren.scene, tgt[i], _MAGENTA, r=0.02)    # actual IK target
            wr.append_data(ren.render())
        wr.close()
        print(f"  wrote {path}  ({len(q)} frames)")


def resolve_config(c):
    for cand in (pathlib.Path(c), g.IK_CONFIG_ROOT / c,
                 g.IK_CONFIG_ROOT / (c if c.endswith(".json") else c + ".json")):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"config not found: {c}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(S.ASIMOV_IK_CONFIG))
    ap.add_argument("--leveling", type=int, default=1)
    ap.add_argument("--clip", default=None)
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--collision", action="store_true",
                    help="make the visual mesh transparent and show collision primitives (green)")
    ap.add_argument("--mesh_alpha", type=float, default=0.12, help="visual-mesh alpha in --collision mode")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no_view", action="store_true")
    ap.add_argument("--dir", default=SRC_DIR, help="clip source directory")
    ap.add_argument("--render", default=None,
                    help="headless: write an .mp4 (single clip) or a dir of mp4s (EGL, no display — for SSH)")
    ap.add_argument("--render_orbit", type=float, default=0.0, help="camera azimuth spin, deg/sec (0 = fixed)")
    ap.add_argument("--render_az", type=float, default=135.0, help="camera azimuth, degrees")
    ap.add_argument("--render_elev", type=float, default=-12.0, help="camera elevation, degrees")
    ap.add_argument("--render_z", type=float, default=0.6, help="camera lookat height (fixed, so float is visible)")
    ap.add_argument("--render_w", type=int, default=640)
    ap.add_argument("--render_h", type=int, default=480)
    a = ap.parse_args()

    cfg = resolve_config(a.config)
    g.params.IK_CONFIG_DICT["smplx"]["asimov"] = cfg
    mr.IK_CONFIG_DICT["smplx"]["asimov"] = cfg
    S.LEVEL_FEET_ENABLED = bool(a.leveling)
    print(f"config   = {cfg}")
    print(f"leveling = {bool(a.leveling)}   overlay = {a.overlay}")

    clips = sorted(glob.glob(f"{a.dir}/*.npz"))
    if a.clip:
        clips = [c for c in clips if a.clip.lower() in os.path.basename(c).lower()]
    if not clips:
        sys.exit(f"no clips match {a.clip!r} in {a.dir}")
    print(f"retargeting {len(clips)} clip(s) ...")

    seqs, xml = [], None
    for c in clips:
        try:
            data, bm, out, h = load_smplh_amass_file(c, BODY_MODELS)
            frames, fps = get_smplx_data_offline_fast(data, bm, out, a.fps)
            gmr = GMR(src_human="smplx", tgt_robot="asimov", actual_human_height=h, verbose=False)
            if a.leveling:
                S._shape_base_tasks(gmr)   # yaw-only base rot + lateral chord pin
                if S.SPINE3_ABOUT_HIPS and frames:
                    S._calibrate_trunk(gmr, frames)   # rigid trunk radii (no scrunch)
            if frames:
                S._init_base_from_human(gmr, frames[0])   # avoid yaw-flipped frame-0 minimum
            pinner = (S._FootPinner(frames)
                      if (a.leveling and S.PIN_STANCE_FOOT_TARGETS and frames) else None)
            raw, skel, tgt = [], [], []
            for f in frames:
                raw.append(S._retarget_frame(gmr, f, pinner=pinner) if a.leveling
                           else gmr.retarget(f).copy())
                if a.overlay:
                    sc = gmr.scale_human_data(f, gmr.human_root_name, gmr.human_scale_table)
                    skel.append({k: v[0].copy() for k, v in sc.items()})            # pre-offset
                    tg = gmr.scaled_human_data                                       # post-offset target
                    tgt.append({k: tg[k][0].copy() for k in ROBOT_LINK_MAP if k in tg})
            raw = np.array(raw)
            # grounding: contact-gated per-frame offset (grounds stance, keeps
            # flight) when a stance is detectable, else the single global offset.
            # `goff` is always per-frame so the overlay shift can track it.
            contact = S._human_contact_mask(frames) if S.CONTACT_GROUNDING else None
            if contact is not None and contact.mean() >= S.MIN_CONTACT_FRAC:
                goff = S._contact_ground_offsets(
                    gmr.xml_file, raw, contact,
                    per_foot=S._per_foot_contact(frames) if S.STABLE_SUPPORT_GROUNDING else None)
            else:
                goff = np.full(len(raw), S._ground_offset_from_geoms(
                    gmr.xml_file, raw, ground_clearance=S.GROUND_CLEARANCE))
            xy0 = raw[0, :2].copy()
            q = raw.copy()
            q[:, 2] -= goff
            q[:, :2] -= xy0
            if a.overlay:  # match the robot's per-frame grounding + recentering
                shifts = [np.array([xy0[0], xy0[1], goff[i]]) for i in range(len(raw))]
                skel = [{k: v - shifts[i] for k, v in d.items()} for i, d in enumerate(skel)]
                tgt = [{k: v - shifts[i] for k, v in d.items()} for i, d in enumerate(tgt)]
            seqs.append((os.path.basename(c)[:-4], q, fps, skel, tgt))
            xml = gmr.xml_file
            print(f"  ok  {os.path.basename(c):42s} {len(q):4d} frames")
        except Exception as e:
            print(f"  ERR {os.path.basename(c)}: {e}")

    if a.render and seqs:
        print(f"rendering {len(seqs)} clip(s) headless (EGL) -> {a.render}")
        render_seqs(seqs, xml, a)
        print("done.")
        return

    if a.no_view or not seqs:
        print(f"done ({len(seqs)} clips retargeted).")
        return

    model = mujoco.MjModel.from_xml_path(xml)
    _style_collision(model, a.mesh_alpha)
    data = mujoco.MjData(model)
    print("\nviewer open — orbit/zoom; close the window or Ctrl-C to stop, then edit the "
          "config and re-run.")
    if a.overlay:
        print("orange = scaled SMPL-H (pre-offset, judge human_scale_table);  "
              "magenta = actual IK target (scaled + pos/rot offsets).")
    print("playing (looping):")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        if a.collision:
            viewer.opt.geomgroup[3] = 1   # collision primitives are group 3 (off by default)
        while viewer.is_running():
            for name, q, fps, skel, tgt in seqs:
                print(f"  ▶ {name}")
                dt = 1.0 / (fps * a.speed)
                for i in range(len(q)):
                    if not viewer.is_running():
                        break
                    data.qpos[:] = q[i]
                    mujoco.mj_forward(model, data)
                    if a.overlay and skel:
                        draw_overlay(viewer.user_scn, skel[i], _ORANGE, reset=True)  # scaled human (pre-offset)
                        draw_overlay(viewer.user_scn, tgt[i], _MAGENTA, r=0.02)      # actual IK target
                    viewer.sync()
                    time.sleep(dt)
                if not viewer.is_running():
                    break


if __name__ == "__main__":
    main()
