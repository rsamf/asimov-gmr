"""Regression test for the arm-raise floating bug.

When the human raises their arms overhead, the shorter robot arm can't reach the
target. With the floating base coupled to the arm IK, the solver lifted the whole
robot off the ground (base rose ~55 cm while the human pelvis moved only ~9 cm).
`DECOUPLE_ARMS_FROM_BASE` solves the arms with the base velocity locked, so the
robot base tracks the human pelvis instead of floating.

Skips cleanly if the SMPL-X body models or the source clip aren't on this machine.
"""
import os
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "scripts"))

from conftest import amass_clip  # noqa: E402

BODY = HERE.parent / "assets" / "body_models" / "smplx"
CLIP = amass_clip("BMLmovi", "Subject_30_F_MoSh", "Subject_30_F_19_poses.npz")

pytestmark = pytest.mark.skipif(
    not (BODY / "SMPLX_NEUTRAL.npz").exists() or not os.path.exists(CLIP),
    reason="needs SMPL-X body models and the source AMASS clip",
)


def _retarget_pelvis_range(decouple):
    """Retarget the arm-raise clip and return (robot pelvis z-range, human pelvis z-range)."""
    import retargeting as g
    from retargeting import motion_retarget as mr
    # this clip's dataset was produced with the tune IK config
    # importing smplx_to_asimov registers the asimov config

    from retargeting import GeneralMotionRetargeting as GMR
    from retargeting.utils.smpl import (
        load_smplh_amass_file, get_smplx_data_offline_fast)
    import smplx_to_asimov as S

    S.DECOUPLE_ARMS_FROM_BASE = decouple
    data, bm, out, h = load_smplh_amass_file(CLIP, str(BODY.parent))
    frames, _ = get_smplx_data_offline_fast(data, bm, out, tgt_fps=30)
    gmr = GMR(src_human="smplx", tgt_robot="asimov", actual_human_height=h, verbose=False)
    qpos = np.array([S._retarget_frame(gmr, fr) for fr in frames])
    robot_range = float(qpos[:, 2].max() - qpos[:, 2].min())
    human_z = data["trans"][:, 2]
    return robot_range, float(human_z.max() - human_z.min())


def test_decoupled_base_tracks_human_pelvis_on_arm_raise():
    robot_range, human_range = _retarget_pelvis_range(decouple=True)
    # the human barely moves vertically raising their arms (~0.09 m); the robot
    # base must stay close to that, not balloon to ~0.55 m as the coupled solve did.
    assert human_range < 0.15, f"unexpected human clip: {human_range:.3f} m"
    assert robot_range < 0.20, (
        f"base still floats: robot pelvis range {robot_range:.3f} m "
        f"vs human {human_range:.3f} m")


CLIP_TURNED = amass_clip("CMU", "36", "36_36_poses.npz")


@pytest.mark.skipif(
    not (BODY / "SMPLX_NEUTRAL.npz").exists() or not os.path.exists(CLIP_TURNED),
    reason="needs SMPL-X body models and the source AMASS clip")
def test_base_init_prevents_yaw_flipped_local_minimum():
    """A clip whose human starts turned away from the model's default heading.

    The decoupled solve removed the arms' front/back-asymmetric pull on the
    base, so the frame-0 lower-body solve can fall into a ~180deg-flipped local
    minimum and warm-starting locks the whole clip there (hip/waist yaws pinned;
    the CMU 36_36 pathology). Initializing the base at the human root target
    before the first solve keeps it in the correct basin.
    """
    import numpy as np
    import mujoco
    from scipy.spatial.transform import Rotation as Rot
    import retargeting as g
    from retargeting import motion_retarget as mr
    # importing smplx_to_asimov registers the asimov config
    from retargeting import GeneralMotionRetargeting as GMR
    from retargeting.utils.smpl import (
        load_smplh_amass_file, get_smplx_data_offline_fast)
    import smplx_to_asimov as S

    data, bm, out, h = load_smplh_amass_file(CLIP_TURNED, str(BODY.parent))
    frames, _ = get_smplx_data_offline_fast(data, bm, out, tgt_fps=30)
    gmr = GMR(src_human="smplx", tgt_robot="asimov", actual_human_height=h, verbose=False)
    S._init_base_from_human(gmr, frames[0])
    for fr in frames[:5]:
        q = S._retarget_frame(gmr, fr)
    m, d = gmr.configuration.model, gmr.configuration.data
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, gmr.robot_root_name)
    Rr = d.xmat[pb].reshape(3, 3)
    Rh = Rot.from_quat(gmr.scaled_human_data[gmr.human_root_name][1],
                       scalar_first=True).as_matrix()
    yr = np.arctan2(Rr[1, 0], Rr[0, 0]); yh = np.arctan2(Rh[1, 0], Rh[0, 0])
    yaw_err = abs(np.degrees(np.arctan2(np.sin(yr - yh), np.cos(yr - yh))))
    # without base init this clip solves ~170deg backwards
    assert yaw_err < 30, f"base still yaw-flipped: {yaw_err:.1f} deg"


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"),
                    reason="slow (retargets the clip twice); set RUN_SLOW=1 to run")
def test_decoupled_beats_coupled():
    """The decoupled solve keeps the base far closer to the ground than the coupled one."""
    decoupled_range, _ = _retarget_pelvis_range(decouple=True)
    coupled_range, _ = _retarget_pelvis_range(decouple=False)
    assert decoupled_range < 0.4 * coupled_range, (
        f"decoupled {decoupled_range:.3f} m not much better than "
        f"coupled {coupled_range:.3f} m")
