"""Retarget a single AMASS SMPL+H clip onto asimov; optional pkl save."""
import argparse, os, pickle, pathlib
import numpy as np
import torch
import mujoco
import mink
from scipy.spatial.transform import Rotation as Rot
from retargeting import GeneralMotionRetargeting as GMR
from retargeting.kinematics_model import KinematicsModel
from retargeting.utils.smpl import (
    load_smplh_amass_file, get_smplx_data_offline_fast,
)

HERE = pathlib.Path(__file__).parent
BODY_MODELS = HERE.parent / "assets" / "body_models"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# The asimov IK config lives WITH the asimov code (NEW/configs). Importing this
# module makes it the active smplx->asimov config for everything downstream —
# batch, viewer, metrics, tests — replacing the per-script config patching that
# every entry point used to repeat.
ASIMOV_IK_CONFIG = HERE.parent / "configs" / "smplx_to_asimov.json"
import retargeting as _g
from retargeting import motion_retarget as _mr
_g.params.IK_CONFIG_DICT["smplx"]["asimov"] = ASIMOV_IK_CONFIG
_mr.IK_CONFIG_DICT["smplx"]["asimov"] = ASIMOV_IK_CONFIG

# Human foot joints whose orientation target we partially flatten.
LEVEL_FEET = ("left_foot", "right_foot")
# When False, use plain GMR retarget (no foot-leveling override) — used by the
# hybrid config, whose devs-style foot handling (pos-offset) doesn't need leveling.
LEVEL_FEET_ENABLED = True
# Fraction of the human foot tilt to keep (1.0 = full human tilt, 0.0 = flat).
# Tracking the full human toe tilt tips the robot foot during heel-strike/toe-off;
# the sole's lowest point then swings and global grounding rests that transient dip
# on the floor, lifting every other frame (~5cm hover). Scaling the tilt toward
# level shrinks those dips (less hover) while keeping foot articulation (toe still
# points on a kick). Combined with hip/knee position weights (config) that raise
# the toe, this lands feet ~2cm off the ground without flattening the motion.
FOOT_TILT_ALPHA = 0.4

# Limit-aware replacement for the constant-alpha shrink above. Measured over the
# corpus, the human's foot targets exceed asimov's ankle ROLL range (+-5.7 deg) on
# 45.7% of frames -- matching the 48% ankle-roll saturation seen in the retargeted
# output -- and its PITCH range (+-20.1 deg) on 15.9%. A constant alpha is wrong at
# both ends: it needlessly flattens the ~54% of frames that were already reachable
# (median roll 5.1 deg -> 2.0 deg, discarding real articulation) while leaving the
# extremes far outside (a 60 deg roll -> 24 deg, still 4x past the stop). Those
# survivors keep the sole tilted onto an edge, which makes the contact-grounding
# reference flip between feet and bobs the whole robot vertically (measured 2.9 cm
# on BioMotionLab_NTroje rub077 0031_rom). Clamping instead is identity where the
# pose is reachable and exact at the boundary where it is not.
# REQUIRES foot rot_w in the config (set to 20): with the old rot_w 0/5 against
# pos_w 100/50 the IK barely tracked foot orientation at all -- the ankles were
# free null-space DOFs drifting into their stops -- so clamping a target nothing
# follows was a measured no-op (ankle-sat 29.7% -> 30.3%).
# MEASURED with foot rot_w 20 (2026-08, ROM/WALK/JUMP/KICK): ankle saturation
# improves on ROM (32.4 -> 23.3%) and WALK (16.3 -> 8.1%) but REGRESSES on KICK
# (7.0 -> 13.6%), p95 sole tilt is flat-to-worse, and the inter-foot floor
# disagreement grows on every clip. Not shipped; revisit only with a fix for the
# kick regression.
FOOT_TILT_CLAMP = False     # see the MEASURED note above: not a win as-is
TILT_LIMIT_MARGIN = 0.9     # target this fraction of the range, so joints don't pin

# The asimov waist is a single yaw joint (no roll/pitch DOF). Tracking spine3's
# full orientation asks the waist for roll/pitch it can't do; that unreachable
# target can tilt the floating base to compromise. When True, the waist target is
# projected to yaw-only (heading) so the IK asks the waist only for what it can
# deliver and stops it pulling the base.
LOCK_WAIST_YAW_ONLY = False
WAIST_HUMAN = ("spine3",)

# Pelvis (floating base) tracks TRANSLATION + YAW only; its pitch/roll are left
# free for spine3 to drive by hinging the whole rigid pelvis+torso at the hips.
# The human bends with two stacked pitch/roll rotations -- spine3 about the pelvis
# (lumbar flex) and the pelvis about the hips (hip hinge). asimov has neither a
# waist pitch/roll joint nor an independent pelvis pitch/roll DOF, so both must
# collapse into ONE rotation on the robot: spine3-about-hips. If the base
# orientation task tracked the human pelvis pitch/roll it would fight spine3 for
# the single base pitch DOF (over-tilt + the waist-kink compromise); if it tracked
# nothing (rot_w=0) the heading would float on spine3 alone and drift on turns. So
# we keep only the yaw column of the base orientation cost: heading + translation
# stay anchored to the human pelvis, pitch/roll follow spine3. The waist_yaw joint
# then naturally represents the human's torso-yaw-relative-to-pelvis (spine twist).
PELVIS_YAW_ONLY = True
# Final-pass base position cost on local x/y (0 disables). Pins the base onto the
# hip->spine3 chord LINE without fighting the along-axis leg-proportion sag; the
# config weight is kept on local z. See _shape_base_tasks.
PELVIS_LATERAL_POS_COST = 100.0

# Collapse the human's two stacked pitch/roll rotations -- spine3-about-pelvis
# (lumbar flexion) and pelvis-about-hips (hip hinge) -- into the ONE the robot
# has: the rigid pelvis+torso rotating about the hip joints. Two identities
# define "spine3 rotates about the hips" and both fail with raw targets:
#   I1 tilt : robot torso z-axis == the CHORD hip_center->spine3. The raw
#             spine3 FRAME embeds lumbar curvature a rigid torso cannot
#             reproduce (tipped ~17 deg back of the chord standing, ~15 deg
#             past it at a deep bend, up to 43 deg off mid-bend when the
#             lumbar flexes before the chord swings).
#   I2 pivot: the base must move rigidly ABOUT THE HIP CENTER, not about the
#             pelvis point (raw tracking swings the robot hip joints ~6 cm
#             off the human hips over a lift clip).
# Fix (all human-side, no robot constants):
#   torso quat <- retilt by rot_between(spine3_target_z, chord)  [kills the
#                 frame artifact; twist about the chord is preserved]
#   base pos   <- hip_c + |pelvis_pos - hip_c| * chord_dir  [the base point is
#                 rigid with the assembly, so it RIDES THE CHORD at its rigid
#                 arm length above the hip center: exact at upright (pelvis
#                 already sits on the vertical chord), full-chord swing about
#                 the hips at a bend]
#   base quat  <- retilt by rot_between(unit(pelvis_pos - hip_c), chord), the
#                 same rigid-assembly rotation its position arm underwent
#                 (geometric, convention-free; only its yaw is costed anyway).
SPINE3_ABOUT_HIPS = True
HIP_HUMAN = ("left_hip", "right_hip")
MIN_CHORD = 0.05   # m: chord shorter than this (degenerate pose) -> keep raw targets
# The trunk targets SCRUNCH as the human bends: (a) SMPL spine3 sits atop a
# curving 3-joint spine, so the raw pelvis->spine3 chord shortens ~26% in a deep
# bend (truly rigid pairs like pelvis->hip are constant to 0.00 std); (b) config
# pos_offsets are applied in the body's LOCAL frame, so they rotate with pose and
# turned the rigid 10 cm pelvis->hip_c arm into 13.6 cm standing -> 6.5 cm bent.
# The rigid robot cannot scrunch, so both radii are CALIBRATED at the clip's
# most-upright frame (offsets frozen in their standing direction) and held
# constant: per frame only the chord DIRECTION is live. See _calibrate_trunk.

# Decouple the floating base from arm IK. The arm position tasks target the
# scaled human wrist/elbow; when the human raises an arm the shorter robot arm
# can't reach, and because the base is a free joint with nothing pinning it, the
# cheapest way for the joint IK to shrink the arm error is to translate the whole
# base upward — the robot floats off the ground with legs dangling (measured:
# human pelvis moves 9 cm while the robot base flies up 55 cm on an arm raise).
# When True, each IK pass is solved in two stages: first the base + lower body
# (pelvis/hips/knees/ankles/waist) with the base free, then the arms with the
# base velocity zeroed — so the arms reach only as far as the arm joints allow
# and can never move the base. Genuine airborne motion (running/jumping) is
# preserved because the base is still driven by the leg/foot tasks. Trade-off:
# overhead-reach targets the robot can't reach stay unreached (larger wrist
# error) instead of being "reached" by lifting the whole robot off the floor.
DECOUPLE_ARMS_FROM_BASE = True
# Robot frame-name substrings whose IK tasks must not move the base.
ARM_FRAME_KEYS = ("shoulder", "elbow", "wrist")
BASE_DOF = 6  # floating-base velocity components (3 translation + 3 rotation)

# Contact-mask grounding (replaces the single global vertical offset). A constant
# offset rests the whole clip's lowest-ever frame on the floor and lets every
# taller frame float; per-frame grounding instead would glue a genuinely airborne
# robot (jump/run flight) to the ground. This follows PBHC (arXiv:2506.12851 v3,
# "Motion Correction based on Contact Mask", Eq. 2-3): a foot is in contact when
# it is both nearly stationary and near the floor; on contact frames we drop the
# robot so its lowest geom sits on the floor; during flight (no foot in contact)
# we HOLD the last contact offset so real air time survives; an EMA smooths the
# offset to remove contact/flight jitter. Contact is detected on the *human* foot
# (its mocap height is a true floor reference; the robot foot height is corrupted
# by the very float we are correcting).
# Ground to the SUPPORT FOOT, committed across a stance, instead of to whichever
# point of the robot is momentarily lowest. A saturated ankle tilts a sole onto its
# edge, and a tilted foot pokes ~1-3 cm lower than a flat one, so the per-frame
# "lowest geom" reference keeps swapping feet and walks the whole body up and down
# (the "hips come up dragging the feet" artifact). Flight is unaffected: with no
# foot planted there is no support foot and the offset is held, exactly as before.
# MEASURED (2026-08) AND REJECTED: this makes the bob WORSE on 2 of 3 clips
# (ROM 6.14 -> 6.85 cm, KICK 3.87 -> 4.41 cm; only WALK improved 2.88 -> 2.32).
# The premise was wrong -- the dominant driver is each sole's own tilt-driven
# variation, not the L<->R reference swap, and `min` over BOTH feet acts as a
# smoother lower envelope than committing to one. It also cost jump air time
# (airborne 40.5 -> 23.8% where the HUMAN is 53.6%), so it moved flight further
# from ground truth. Kept for reference; do not enable without a new mechanism.
STABLE_SUPPORT_GROUNDING = False

CONTACT_GROUNDING = True
CONTACT_EPS_VEL = 0.010     # m/frame: human-foot displacement below this is "stationary"
CONTACT_EPS_HEIGHT = 0.05   # m: human foot within this of its clip-min height is "down"
GROUND_EMA = 0.15           # EMA weight on the per-frame offset (bidirectional)
MIN_CONTACT_FRAC = 0.05     # if a clip registers contact on fewer frames than this,
                            # fall back to the global offset (no reliable stance to key on)
HUMAN_FEET = ("left_foot", "right_foot")
# Pin each foot's XY position target to its stance anchor while the human foot is
# detected planted, so mocap/SMPL foot creep (measured 2-6 cm per stance) doesn't
# retarget into foot slide. POSITION-ONLY (orientation stays live, so pivoting on a
# grounded foot is preserved). Genuine grounded movement is respected two ways:
# motion fast enough to break the contact velocity threshold releases immediately,
# and slow deliberate slides release once the live target drifts PIN_MAX_DRIFT from
# the anchor. Releases blend over PIN_RELEASE_BLEND frames so there is no pop.
PIN_STANCE_FOOT_TARGETS = True
PIN_MAX_DRIFT = 0.10        # m: live target farther than this from the anchor while
                            # "planted" = intentional grounded slide -> release
PIN_RELEASE_BLEND = 5       # frames to blend from the anchor back to the live target
# Pin the planted foot's Z target as well as XY. The z-target wander during a
# stance (root-leak + in-tolerance human foot motion) is the DOMINANT source of
# the grounding bob -- see _FootPinner docstring for the measured decomposition.
# REJECTED BY EYEBALL (2026-08-09): with the z-pin + z-stiff solve the robot
# visibly raises both feet in places and jitters -- the numeric wins (wander
# p95 -20-30%) did not survive visual review. v5 behavior (z live, isotropic
# foot cost) is the shipped reference; the residual bob is accepted as the
# conservation-bounded tradeoff. Keep these OFF.
PIN_FOOT_Z = False
# The pin engages at contact ONSET = heel-strike, often the HIGHEST foot z of the
# stance; anchoring z there makes the z-stiff solve hold the foot high (measured:
# a walking clip went 0 -> 5.3% airborne). Settle the z-anchor to the MINIMUM of
# the live z over the stance's first frames instead.
PIN_Z_SETTLE = 6
# DO NOT "close gaps" in the contact mask: running flight at 30 fps is only 4-6
# frames of no-contact, indistinguishable from detection flicker by length alone.
# A 5-frame hole-filler was tried (2026-08) and would have silently deleted
# genuine running flight -- the mask's short gaps on KIT walking_run measured
# 3.4-7.9 cm of real human foot lift.
# Per-axis z boost on the foot POSITION cost (mink costs are per-axis). The pinned
# z target is only worth what the solver pays for it: at the config's isotropic
# weight the other tasks trade the planted foot's height away by 2-3 cm within a
# stance (the dominant grounding-bob source); a stiff z axis forces the solve to
# hold it, so the floor reference stays put. 0 disables.
FOOT_Z_COST = 0.0           # off with PIN_FOOT_Z (see rejection note above)

# Hard floor: the lowest robot geom (incl. visual mesh vertices) is guaranteed to
# sit at least this far ABOVE the ground on every frame, so RSI (reference-state
# init) never spawns the robot partly underground. Strictly positive (not 0) as a
# margin against mesh/numeric precision; kept tiny so a foot in contact still reads
# as on-the-ground for the contact-tracking reward. The EMA above can lag the true
# lowest into slight penetration; this clamp removes it (one-sided: only raises,
# never lowers, so genuine flight is untouched).
GROUND_CLEARANCE = 0.001


def _partial_level_quat(quat_wxyz, alpha):
    """Scale a foot orientation toward level: keep heading (yaw), shrink pitch/roll."""
    Rt = Rot.from_quat(quat_wxyz, scalar_first=True)
    M = Rt.as_matrix()
    Rlevel = Rot.from_euler("z", np.arctan2(M[1, 0], M[0, 0]))
    tilt = (Rlevel.inv() * Rt).as_rotvec()
    return (Rlevel * Rot.from_rotvec(alpha * tilt)).as_quat(scalar_first=True)


def _ankle_tilt_limits(model, margin=None):
    """(roll_max, pitch_max) radians the ankle can actually deliver, from the model.

    Read from the MuJoCo joint ranges (never hardcoded: asimov's roll is a mere
    +-5.7 deg while pitch is +-20.1 deg) and shrunk by `margin` so targets land
    just INSIDE the stop -- a target exactly at the limit leaves the joint pinned,
    which makes the reference's joint velocities chatter."""
    if margin is None:
        margin = TILT_LIMIT_MARGIN
    out = []
    for axis in ("roll", "pitch"):
        rng = []
        for side in ("left", "right"):
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_ankle_{axis}_joint")
            if j >= 0:
                rng.append(min(abs(model.jnt_range[j][0]), abs(model.jnt_range[j][1])))
        out.append(margin * (min(rng) if rng else np.pi))
    return out[0], out[1]


def _clamp_tilt_quat(quat_wxyz, roll_max, pitch_max):
    """Project a foot orientation onto what the ankle can reach (see FOOT_TILT_CLAMP).

    Heading (yaw) is preserved exactly; the residual tilt is decomposed on the
    ankle's own axes -- roll about x, pitch about y -- and each is CLAMPED to its
    range. Unlike a constant-alpha shrink this is identity whenever the human's
    foot is already reachable (~54% of frames for roll, 84% for pitch), so real
    articulation survives, while genuinely out-of-range frames land exactly on the
    boundary instead of a fraction of the way past it."""
    Rt = Rot.from_quat(quat_wxyz, scalar_first=True)
    M = Rt.as_matrix()
    Rlevel = Rot.from_euler("z", np.arctan2(M[1, 0], M[0, 0]))
    roll, pitch, resid = (Rlevel.inv() * Rt).as_euler("xyz")
    roll = float(np.clip(roll, -roll_max, roll_max))
    pitch = float(np.clip(pitch, -pitch_max, pitch_max))
    return (Rlevel * Rot.from_euler("xyz", [roll, pitch, resid])).as_quat(scalar_first=True)


def _rot_between(a, b):
    """Minimal world-frame rotation R with R @ a = b (a, b unit vectors)."""
    c = np.cross(a, b)
    s = float(np.linalg.norm(c))
    d = float(np.dot(a, b))
    if s < 1e-9:
        if d >= 0.0:
            return Rot.identity()
        p = np.cross(a, (1.0, 0.0, 0.0))          # antiparallel: 180 deg about
        if np.linalg.norm(p) < 1e-6:              # any axis perpendicular to a
            p = np.cross(a, (0.0, 1.0, 0.0))
        return Rot.from_rotvec(np.pi * p / np.linalg.norm(p))
    return Rot.from_rotvec((c / s) * np.arctan2(s, d))


def _clean_scaled(gmr, frame):
    """Scaled human joints WITHOUT the per-task pos_offsets — clean anatomy.

    gmr.scaled_human_data holds TASK TARGETS: scale_human_data output plus each
    tracked body's config pos_offset rotated into its (rot_offset) target frame.
    Those offsets place ROBOT LINK ORIGINS (e.g. the hip targets are pushed
    [0.05, ±0.03, -0.03] to land on hip_roll_link) and must not be read as
    anatomy: a 5 cm forward push of the hip centers tilts the hip->spine3 chord
    ~6 deg BACKWARD, which the torso then faithfully tracks (the "robot leans
    back" report). Geometry (chord direction, rigid radii) must come from here;
    task-space positions (sd) are still the right anchor for target PLACEMENT."""
    return gmr.scale_human_data(frame, gmr.human_root_name, gmr.human_scale_table)


def _calibrate_trunk(gmr, frames, stride=3):
    """Freeze the trunk radii at the clip's most-upright frame (see the scrunch
    note at SPINE3_ABOUT_HIPS): stashes gmr._trunk_calib = (arm0, r0) where
    arm0 = |pelvis - hip_c| (base radius) and r0 = |spine3 - hip_c| (trunk
    chord), both from CLEAN scaled anatomy (offset-free, so arm0 is the truly
    rigid pelvis-bone radius). Call once per clip after GMR init; without it
    _spine3_about_hips falls back to the live per-frame lengths."""
    best = None
    for fr in frames[::max(1, stride)]:
        sc = _clean_scaled(gmr, fr)
        if "spine3" not in sc or not all(k in sc for k in HIP_HUMAN):
            continue
        hip_c = 0.5 * (np.asarray(sc[HIP_HUMAN[0]][0]) + np.asarray(sc[HIP_HUMAN[1]][0]))
        chord = np.asarray(sc["spine3"][0]) - hip_c
        n = float(np.linalg.norm(chord))
        arm = float(np.linalg.norm(np.asarray(sc[gmr.human_root_name][0]) - hip_c))
        if n < MIN_CHORD or arm < 1e-6:
            continue
        lean = np.arccos(np.clip(chord[2] / n, -1.0, 1.0))
        if best is None or lean < best[0]:
            best = (lean, arm, n)
    if best is not None:
        gmr._trunk_calib = (best[1], best[2])


def _spine3_about_hips(gmr, frame):
    """Resynthesize the torso + base targets so the rigid pelvis+torso rotates
    about the HIP CENTER to follow the hip_center->spine3 chord (identities I1 +
    I2, see SPINE3_ABOUT_HIPS). The chord DIRECTION and rigid radii come from
    CLEAN scaled anatomy (_clean_scaled: task pos_offsets excluded — they place
    robot link origins and would tilt the chord ~6 deg backward); the target
    PLACEMENT anchors at the task-space hip center so the base stays consistent
    with where the leg tasks pull. Radii come from _calibrate_trunk when
    available, so the trunk cannot scrunch. Writes the synthesized targets back
    into gmr.scaled_human_data so the viewer overlay and task-error readouts
    show what the IK actually tracks. Call after gmr.update_targets()."""
    sd = gmr.scaled_human_data
    root = gmr.human_root_name
    if "spine3" not in sd or not all(k in sd for k in HIP_HUMAN):
        return
    sc = _clean_scaled(gmr, frame)
    hip_c_clean = 0.5 * (np.asarray(sc[HIP_HUMAN[0]][0]) + np.asarray(sc[HIP_HUMAN[1]][0]))
    chord = np.asarray(sc["spine3"][0]) - hip_c_clean
    n = float(np.linalg.norm(chord))
    if n < MIN_CHORD:
        return
    vhat = chord / n                                          # anatomy: clean direction
    arm = np.asarray(sc[root][0]) - hip_c_clean
    arm_len = float(np.linalg.norm(arm))
    if arm_len < 1e-6:
        return
    s_quat = sd["spine3"][1]
    Rs = Rot.from_quat(s_quat, scalar_first=True)
    q_torso = (_rot_between(Rs.apply((0.0, 0.0, 1.0)), vhat) * Rs
               ).as_quat(scalar_first=True)                   # I1: torso z == chord
    hip_c = 0.5 * (np.asarray(sd[HIP_HUMAN[0]][0]) + np.asarray(sd[HIP_HUMAN[1]][0]))
    arm0, r0 = getattr(gmr, "_trunk_calib", (arm_len, n))     # rigid radii (no scrunch)
    base_pos = hip_c + arm0 * vhat                            # I2: base rides the chord
    s3_pos = hip_c + r0 * vhat                                # trunk length held rigid
    dR = _rot_between(arm / arm_len, vhat)                    # the arm's rigid rotation
    q_base = (dR * Rot.from_quat(sd[root][1], scalar_first=True)).as_quat(scalar_first=True)
    _set_target(gmr, "spine3", s3_pos, q_torso)
    _set_target(gmr, root, base_pos, q_base)
    sd["spine3"] = (s3_pos, q_torso)
    sd[root] = (base_pos, q_base)


def _solve(gmr):
    """Run GMR's IK solve over both match tables (targets already set)."""
    for use, tasks, err in (
        (gmr.use_ik_match_table1, gmr.tasks1, gmr.error1),
        (gmr.use_ik_match_table2, gmr.tasks2, gmr.error2),
    ):
        if not use:
            continue
        dt = gmr.configuration.model.opt.timestep
        cur = err()
        v = mink.solve_ik(gmr.configuration, tasks, dt, gmr.solver, gmr.damping, gmr.ik_limits)
        gmr.configuration.integrate_inplace(v, dt)
        nxt = err()
        it = 0
        while cur - nxt > 0.001 and it < gmr.max_iter:
            cur = nxt
            v = mink.solve_ik(gmr.configuration, tasks, dt, gmr.solver, gmr.damping, gmr.ik_limits)
            gmr.configuration.integrate_inplace(v, dt)
            nxt = err()
            it += 1
    return gmr.configuration.data.qpos.copy()


def _init_base_from_human(gmr, frame):
    """Start the floating base at the human root target before the first solve.

    With DECOUPLE_ARMS_FROM_BASE the base is driven by the lower body only,
    which is nearly front/back symmetric — on clips whose human starts turned
    away from the model's default heading, the frame-0 solve can fall into a
    ~180 deg yaw-flipped local minimum and per-frame warm-starting locks the
    whole clip there (hip/waist yaws pinned at their stops; CMU 36_36). The
    coupled solve escaped via the arms' asymmetric pull, which decoupling
    removed. Initializing the base in the correct basin restores the coupled
    solve's quality. Call once per clip, before the first _retarget_frame.
    """
    gmr.update_targets(frame)
    pos, quat = gmr.scaled_human_data[gmr.human_root_name]
    q = gmr.configuration.data.qpos.copy()
    q[:3] = pos
    q[3:7] = quat                       # wxyz, matching the free joint
    gmr.configuration.update(q)


def _shape_base_tasks(gmr):
    """Reshape the base (pelvis_link) task costs for the underactuated pelvis.
    mink FrameTask costs are per-axis body twists in the frame's local axes, so:

    Orientation -> YAW ONLY [0, 0, w] (PELVIS_YAW_ONLY): local z = world yaw
    while upright -- walking/turning, exactly when heading matters. Pitch/roll
    get zero cost and are free for spine3 to drive. The config rot_w is reused
    as the yaw weight.

    Position (final pass only) -> LATERAL PIN [lat, lat, w_cfg]
    (PELVIS_LATERAL_POS_COST): local z runs along the torso axis (== the chord
    when tracking), so a high local-x/y cost pins the base ONTO the
    hip_center->spine3 chord line while the soft along-axis weight leaves the
    leg-proportion sag unfought. Measured on the lift clip: base-off-chord
    9.3 -> 1.8 cm max with feet still pinned; an isotropic weight raise instead
    trades tilt + foot pinning for it (zero-sum).

    Call once per clip, right after GMR init and before the first solve; costs
    persist (update_targets only rewrites targets)."""
    if PELVIS_YAW_ONLY:
        for tbl in (gmr.human_body_to_task1, gmr.human_body_to_task2):
            task = tbl.get(gmr.human_root_name)
            if task is not None:
                w = float(np.atleast_1d(task.cost[3:])[-1])   # config rot_w -> yaw weight
                task.set_orientation_cost([0.0, 0.0, w])
    if PELVIS_LATERAL_POS_COST:
        task = gmr.human_body_to_task2.get(gmr.human_root_name)
        if task is not None:
            w = float(np.atleast_1d(task.cost[:3])[-1])       # config pos weight stays on local z
            lat = max(PELVIS_LATERAL_POS_COST, w)
            task.set_position_cost([lat, lat, w])
    if FOOT_Z_COST:
        # Stiff z on the feet so the pinned stance height is actually held (see
        # FOOT_Z_COST). XY keeps the config weight; asimov foot frames are world-
        # aligned, so the local z axis is vertical whenever the sole is near level.
        for tbl in (gmr.human_body_to_task1, gmr.human_body_to_task2):
            for k in LEVEL_FEET:
                task = tbl.get(k)
                if task is not None:
                    w = float(np.atleast_1d(task.cost[:3])[-1])
                    task.set_position_cost([w, w, max(FOOT_Z_COST, w)])


def _is_arm_task(task):
    return any(k in task.frame_name for k in ARM_FRAME_KEYS)


def _solve_task_group(gmr, tasks, lock_base):
    """Run the IK convergence loop over one subset of tasks. When `lock_base`,
    the floating-base velocity is zeroed each step so these tasks (the arms)
    move only their own joints and never translate/rotate the base."""
    if not tasks:
        return
    dt = gmr.configuration.model.opt.timestep
    group_err = lambda: np.linalg.norm(
        [t.compute_error(gmr.configuration) for t in tasks])

    def step():
        v = mink.solve_ik(gmr.configuration, tasks, dt, gmr.solver,
                          gmr.damping, gmr.ik_limits)
        if lock_base:
            v = v.copy()
            v[:BASE_DOF] = 0.0
        gmr.configuration.integrate_inplace(v, dt)

    cur = group_err(); step(); nxt = group_err(); it = 0
    while cur - nxt > 0.001 and it < gmr.max_iter:
        cur = nxt; step(); nxt = group_err(); it += 1


def _solve_decoupled(gmr):
    """Like _solve, but per pass solve base+lower-body first (base free), then
    the arms with the base locked — so limb IK cannot move the floating base."""
    for use, tasks in ((gmr.use_ik_match_table1, gmr.tasks1),
                       (gmr.use_ik_match_table2, gmr.tasks2)):
        if not use:
            continue
        _solve_task_group(gmr, [t for t in tasks if not _is_arm_task(t)], lock_base=False)
        _solve_task_group(gmr, [t for t in tasks if _is_arm_task(t)], lock_base=True)
    return gmr.configuration.data.qpos.copy()


def _waist_yaw_only(child_quat_wxyz, parent_quat_wxyz):
    """Project the waist target to what a single yaw joint on the parent can reach:
    keep the parent's (pelvis) roll/pitch, keep only the child's yaw *about the
    parent's up-axis*. Drops the torso roll/pitch the asimov waist can't do, WITHOUT
    forcing the torso world-upright (it still leans with the pelvis)."""
    Rp = Rot.from_quat(parent_quat_wxyz, scalar_first=True)
    Rs = Rot.from_quat(child_quat_wxyz, scalar_first=True)
    M = (Rp.inv() * Rs).as_matrix()                       # child expressed in parent frame
    yaw = np.arctan2(M[1, 0], M[0, 0])                    # yaw about parent local z
    return (Rp * Rot.from_euler("z", yaw)).as_quat(scalar_first=True)


def _set_target(gmr, hname, pos, quat_wxyz):
    target = mink.SE3.from_rotation_and_translation(mink.SO3(quat_wxyz), pos)
    for tbl in (gmr.human_body_to_task1, gmr.human_body_to_task2):
        if hname in tbl:
            tbl[hname].set_target(target)


def _retarget_frame(gmr, frame, alpha=FOOT_TILT_ALPHA, lock_waist=None, pinner=None):
    """Retarget one frame. Feet orientation targets are projected onto what the
    ankle can reach (FOOT_TILT_CLAMP; else scaled toward level by `alpha`); if
    `pinner`, planted feet's XY position targets are held at their stance anchor
    (see _FootPinner); if `lock_waist`, the waist target keeps only pelvis-relative
    yaw (asimov waist can only yaw); with SPINE3_ABOUT_HIPS the torso/base targets
    are resynthesized to hinge about the hip centers."""
    if lock_waist is None:
        lock_waist = LOCK_WAIST_YAW_ONLY
    gmr.update_targets(frame)
    if FOOT_TILT_CLAMP or alpha < 1.0 or pinner is not None:
        limits = (_ankle_tilt_limits(gmr.configuration.model) if FOOT_TILT_CLAMP else None)
        for hname in LEVEL_FEET:
            pos, quat = gmr.scaled_human_data[hname]
            if pinner is not None:
                pos = pinner.pos(hname, pos)
            if FOOT_TILT_CLAMP:
                quat = _clamp_tilt_quat(quat, *limits)
            elif alpha < 1.0:
                quat = _partial_level_quat(quat, alpha)
            _set_target(gmr, hname, pos, quat)
    if lock_waist:
        pelvis_quat = gmr.scaled_human_data[gmr.human_root_name][1]
        for hname in WAIST_HUMAN:
            if hname in gmr.scaled_human_data:
                pos, quat = gmr.scaled_human_data[hname]
                _set_target(gmr, hname, pos, _waist_yaw_only(quat, pelvis_quat))
    if SPINE3_ABOUT_HIPS:
        _spine3_about_hips(gmr, frame)   # torso tilt <- clean hip chord; base swings about the hips
    return _solve_decoupled(gmr) if DECOUPLE_ARMS_FROM_BASE else _solve(gmr)


def _geom_lowest_z(model, data, g, mesh_corners):
    """Exact lowest world-z of a single geom, by type.

    The IK targets body *origins*, but asimov's ankle_roll_link origin sits
    ~3.4 cm above the foot sole, so grounding on origins sinks the foot.
    Bounding-box corners over-estimate the extent of a rotated mesh (they sit
    below the true surface) and lift the robot, so we use true geometry per
    type instead.
    """
    gtype = model.geom_type[g]
    p = data.geom_xpos[g]
    R = data.geom_xmat[g].reshape(3, 3)
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return p[2] - model.geom_size[g, 0]
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        # two hemisphere centers along local z, each minus the radius
        r, half = model.geom_size[g, 0], model.geom_size[g, 1]
        ends = np.array([[0, 0, half], [0, 0, -half]]) @ R.T + p
        return ends[:, 2].min() - r
    if gtype == mujoco.mjtGeom.mjGEOM_MESH and model.geom_dataid[g] >= 0:
        dataid = model.geom_dataid[g]
        a = model.mesh_vertadr[dataid]
        verts = model.mesh_vert[a:a + model.mesh_vertnum[dataid]]
        return (verts @ R.T + p)[:, 2].min()
    # box: halfsize corners are exact; other types: conservative AABB corners
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        c, hs = np.zeros(3), model.geom_size[g]
    else:
        c, hs = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
    return ((c + mesh_corners * hs) @ R.T + p)[:, 2].min()


def _ground_offset_from_geoms(xml_file, qpos_wxyz, ground_clearance=0.0):
    """Lowest robot geometry z over the whole clip (visual + collision).

    Grounds on the true lowest point of *all* robot geometry, computed exactly
    per geom type. Including the visual foot mesh (not just the collision
    spheres, which don't cover the toe/heel tips) guarantees nothing visible
    clips through the floor; using exact geometry (mesh vertices, sphere
    center-minus-radius) instead of bounding-box corners avoids lifting the
    robot off the ground.
    """
    model = mujoco.MjModel.from_xml_path(str(xml_file))
    data = mujoco.MjData(model)
    PLANE = mujoco.mjtGeom.mjGEOM_PLANE
    # all robot geoms (exclude worldbody / ground plane)
    gids = [g for g in range(model.ngeom)
            if model.geom_bodyid[g] != 0 and model.geom_type[g] != PLANE]
    corners = np.array([[sx, sy, sz] for sx in (-1, 1)
                        for sy in (-1, 1) for sz in (-1, 1)], dtype=float)

    lowest = np.inf
    for q in qpos_wxyz:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for g in gids:
            z = _geom_lowest_z(model, data, g, corners)
            if z < lowest:
                lowest = z
    return lowest - ground_clearance


def _per_foot_contact(frames, eps_vel=CONTACT_EPS_VEL, eps_height=CONTACT_EPS_HEIGHT):
    """Per-foot, per-frame contact masks from the human motion (PBHC Eq. 2).

    A foot is in contact when it is both nearly stationary (small frame-to-frame
    displacement) and near the floor (within `eps_height` of its own lowest point
    over the clip). Detected on the human foot because its mocap height is an
    uncorrupted floor reference.
    """
    out = {}
    for name in HUMAN_FEET:
        foot = np.array([f[name][0] for f in frames])
        v2 = np.sum(np.diff(foot, axis=0) ** 2, axis=1)
        # duplicate the last real velocity for the final frame (a 0-pad would
        # mark a moving foot "stationary" on its last frame)
        vel2 = np.concatenate([v2, v2[-1:]]) if len(v2) else np.zeros(1)
        down = foot[:, 2] < foot[:, 2].min() + eps_height
        out[name] = (vel2 < eps_vel ** 2) & down
    return out


def _human_contact_mask(frames, eps_vel=CONTACT_EPS_VEL, eps_height=CONTACT_EPS_HEIGHT):
    """Per-frame contact mask: either foot in contact (see _per_foot_contact)."""
    mask = np.zeros(len(frames), dtype=bool)
    for m in _per_foot_contact(frames, eps_vel, eps_height).values():
        mask |= m
    return mask


class _FootPinner:
    """Hold each foot's position target at its stance anchor while planted.

    XY: pinned (kills mocap creep -> foot slide). Z: pinned when PIN_FOOT_Z
    (default), because the planted foot's z TARGET moves 1-3.5 cm within a
    stance -- root-relative scaling leaks ~(1-foot_scale) of the human ROOT's
    own bobbing into the foot target, plus the human foot's micro-motion inside
    the contact mask's 5 cm tolerance. The IK tracks that wander at weight 100,
    the floor reference follows the foot, and per-frame grounding re-datums the
    whole body: the "hips rise dragging the feet" bob. Decomposed on ROM/WALK/
    KICK stances: raw foot-height drift median 2.0-2.9 cm vs sole-tilt term
    0.35-0.61 cm -- the z-target wander dominates, so freezing z at the stance
    anchor removes the bob at its source. Orientation stays live (pivoting on a
    grounded foot is preserved).

    The pin engages at contact onset, anchoring at the live target, and releases
    when: (a) contact ends (fast motion breaks the velocity threshold), or
    (b) the live target drifts more than `max_drift` (XY) from the anchor while
    still "planted" — a slow, deliberate grounded slide we must not fight. Every
    release blends anchor->live over `blend` frames to avoid pops.
    Call `pos(foot, live)` exactly once per foot per frame, in frame order.
    """

    def __init__(self, frames, max_drift=None, blend=None):
        self.masks = _per_foot_contact(frames)
        self.n = len(frames)
        self.max_drift = PIN_MAX_DRIFT if max_drift is None else max_drift
        self.blend = PIN_RELEASE_BLEND if blend is None else blend
        self.idx = {f: 0 for f in self.masks}
        self.anchor = {f: None for f in self.masks}
        self.rel_left = {f: 0 for f in self.masks}
        self.settle_left = {f: 0 for f in self.masks}
        self.rel_from = {f: None for f in self.masks}

    def _release(self, foot):
        self.rel_from[foot] = self.anchor[foot].copy()
        self.rel_left[foot] = self.blend
        self.anchor[foot] = None

    def pos(self, foot, live):
        i = self.idx[foot]
        self.idx[foot] = i + 1
        live = np.asarray(live, dtype=float)
        planted = bool(self.masks[foot][i]) if i < self.n else False
        a = self.anchor[foot]
        if planted and a is None and self.rel_left[foot] == 0:
            self.anchor[foot] = live.copy()          # engage at the live target
            self.settle_left[foot] = PIN_Z_SETTLE    # z-anchor settles to stance min
            return live
        if a is not None:
            if not planted:
                self._release(foot)                  # lift-off
            elif np.linalg.norm(live[:2] - a[:2]) > self.max_drift:
                self._release(foot)                  # deliberate grounded slide
            else:
                out = live.copy()
                out[:2] = a[:2]                      # XY pinned
                if PIN_FOOT_Z:
                    if self.settle_left[foot] > 0:   # settle: adopt the stance's low z
                        self.settle_left[foot] -= 1
                        if live[2] < a[2]:
                            a[2] = live[2]
                    out[2] = a[2]                    # Z pinned too (see class doc)
                return out
        if self.rel_left[foot] > 0:                  # blending back to live
            t = 1.0 - self.rel_left[foot] / float(self.blend + 1)
            self.rel_left[foot] -= 1
            out = live.copy()
            out[:2] = (1.0 - t) * self.rel_from[foot][:2] + t * live[:2]
            return out
        return live


def _contact_ground_offsets(xml_file, qpos_wxyz, contact, ema=GROUND_EMA, per_foot=None):
    """Per-frame vertical offsets that ground the robot on contact frames (PBHC Eq. 3).

    On a contact frame the offset is the floor reference (see below); during flight
    the last contact offset is HELD so genuine air time survives; the sequence is
    EMA-smoothed (bidirectional, zero-phase) to remove contact/flight jitter, then
    clamped so nothing ever penetrates. Returns an array to subtract from root z.

    Floor reference, with STABLE_SUPPORT_GROUNDING and `per_foot` masks: the SUPPORT
    FOOT's lowest point, committed for the whole stance (hysteresis: while the current
    support foot is still planted we keep it, even if the other foot dips lower).
    Otherwise: the lowest point anywhere on the robot, re-picked every frame -- which
    silently swaps feet whenever a saturated ankle tilts one sole onto its edge, since
    a tilted foot pokes lower than a flat one. Measured over 247 clips, the two feet
    disagree about the floor by 1.1 cm typically and 3.2 cm at p95, and the reference
    swapped on ~4% of frames, walking the whole body up and down with it.
    """
    model = mujoco.MjModel.from_xml_path(str(xml_file))
    data = mujoco.MjData(model)
    PLANE = mujoco.mjtGeom.mjGEOM_PLANE
    gids = [g for g in range(model.ngeom)
            if model.geom_bodyid[g] != 0 and model.geom_type[g] != PLANE]
    foot_gids = {}
    for hname, link in (("left_foot", "left_ankle_roll_link"),
                        ("right_foot", "right_ankle_roll_link")):
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
        foot_gids[hname] = [g for g in gids if model.geom_bodyid[g] == b]
    corners = np.array([[sx, sy, sz] for sx in (-1, 1)
                        for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    n = len(qpos_wxyz)
    lowest = np.empty(n)
    foot_low = {k: np.empty(n) for k in foot_gids}
    for i, q in enumerate(qpos_wxyz):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        lowest[i] = min(_geom_lowest_z(model, data, g, corners) for g in gids)
        for k, gl in foot_gids.items():
            foot_low[k][i] = (min(_geom_lowest_z(model, data, g, corners) for g in gl)
                              if gl else lowest[i])

    use_support = (STABLE_SUPPORT_GROUNDING and per_foot is not None
                   and all(k in per_foot for k in foot_gids))
    off = np.empty(n)
    last = lowest[int(np.argmax(contact))]   # first contact frame (argmax of bool)
    support = None
    for i in range(n):
        if use_support:
            planted = [k for k in foot_gids if per_foot[k][i]]
            if planted:
                # keep the committed support foot while it is still bearing weight;
                # only hand over when it lifts (or at the first stance of the clip)
                if support not in planted:
                    support = min(planted, key=lambda k: foot_low[k][i])
                last = foot_low[support][i]
            else:
                support = None               # flight: hold the last offset
        elif contact[i]:
            last = lowest[i]
        off[i] = last
    # bidirectional EMA: forward then backward, so the smoothing has no time lag
    for rng in (range(n), range(n - 1, -1, -1)):
        acc = off[rng[0]]
        for i in rng:
            acc = ema * off[i] + (1 - ema) * acc
            off[i] = acc
    # hard floor: never drop a frame below GROUND_CLEARANCE. Since the grounded
    # lowest is (lowest - off), capping off at (lowest - clearance) guarantees
    # lowest - off >= clearance on every frame. One-sided (only reduces the drop),
    # so airborne frames whose EMA offset is already smaller are untouched.
    off = np.minimum(off, lowest - GROUND_CLEARANCE)
    return off


def retarget_clip(smplh_file, save_pkl=None, tgt_fps=30):
    data, bm, out, height = load_smplh_amass_file(str(smplh_file), str(BODY_MODELS))
    frames, fps = get_smplx_data_offline_fast(data, bm, out, tgt_fps=tgt_fps)
    gmr = GMR(src_human="smplx", tgt_robot="asimov",
              actual_human_height=height, verbose=False)
    _shape_base_tasks(gmr)                       # yaw-only base rot + lateral chord pin
    if frames:
        if SPINE3_ABOUT_HIPS:
            _calibrate_trunk(gmr, frames)       # rigid trunk radii (no scrunch)
        _init_base_from_human(gmr, frames[0])   # avoid the yaw-flipped frame-0 minimum
    if LEVEL_FEET_ENABLED:
        pinner = _FootPinner(frames) if (PIN_STANCE_FOOT_TARGETS and frames) else None
        qpos = np.array([_retarget_frame(gmr, fr, pinner=pinner) for fr in frames])
    else:
        qpos = np.array([gmr.retarget(fr).copy() for fr in frames])

    root_pos = qpos[:, :3]
    root_rot = qpos[:, 3:7].copy()
    root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]  # wxyz -> xyzw
    dof_pos = qpos[:, 7:]
    n = root_pos.shape[0]

    km = KinematicsModel(gmr.xml_file, device=DEVICE)
    # height adjust: put the lowest robot *geometry* (foot sole) on the ground,
    # not the lowest body origin (which would sink the foot mesh ~3.7cm).
    contact = _human_contact_mask(frames) if CONTACT_GROUNDING else None
    if contact is not None and contact.mean() >= MIN_CONTACT_FRAC:
        # contact-gated per-frame grounding: grounds stance frames, keeps flight
        goff = _contact_ground_offsets(
            gmr.xml_file, qpos, contact,
            per_foot=_per_foot_contact(frames) if STABLE_SUPPORT_GROUNDING else None)
        root_pos[:, 2] -= goff
    else:
        # no reliable stance to key on (or disabled) -> single global offset
        root_pos[:, 2] -= _ground_offset_from_geoms(gmr.xml_file, qpos,
                                                    ground_clearance=GROUND_CLEARANCE)
    root_pos[:, :2] -= root_pos[0, :2]  # origin offset

    fk_r = torch.zeros((n, 3), device=DEVICE)
    fk_q = torch.zeros((n, 4), device=DEVICE); fk_q[:, -1] = 1.0
    local_body_pos, _ = km.forward_kinematics(
        fk_r, fk_q, torch.from_numpy(dof_pos).to(DEVICE, torch.float))

    motion_data = {
        "fps": fps, "root_pos": root_pos, "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos.detach().cpu().numpy(),
        "link_body_list": km.body_names,
    }
    if save_pkl is not None:
        save_dir = os.path.dirname(save_pkl)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_pkl, "wb") as fh:
            pickle.dump(motion_data, fh)
    return motion_data, fps

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smplx_file", required=True)
    ap.add_argument("--save_path", default=None)
    a = ap.parse_args()
    _, fps = retarget_clip(a.smplx_file, a.save_path)
    print("done, fps:", fps)
