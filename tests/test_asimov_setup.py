import glob

import pytest

from conftest import (BODY_MODELS, amass_clip, requires_amass, requires_robot,
                      requires_smplx)
from retargeting import GeneralMotionRetargeting as GMR
from retargeting.utils.smpl import (
    load_smplh_amass_file, get_smplx_data_offline_fast,
)

AMASS = amass_clip("ACCAD", "Male2Walking_c3d", "*.npz")


@requires_amass
@requires_smplx
@requires_robot
def test_asimov_retarget_one_frame():
    matches = sorted(glob.glob(AMASS))
    if not matches:
        pytest.skip(f"no clips at {AMASS}")
    f = matches[0]
    data, bm, out, h = load_smplh_amass_file(f, BODY_MODELS)
    frames, fps = get_smplx_data_offline_fast(data, bm, out, tgt_fps=30)
    gmr = GMR(src_human="smplx", tgt_robot="asimov", actual_human_height=h, verbose=False)
    qpos = gmr.retarget(frames[0])
    assert qpos.shape[0] == 32   # 7 base + 25 joints
