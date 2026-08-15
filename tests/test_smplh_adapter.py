import glob

import numpy as np
import pytest

from conftest import BODY_MODELS, amass_clip, requires_amass, requires_smplx
from retargeting.utils.smpl import (
    load_smplh_amass_file,
    get_smplx_data_offline_fast,
)

AMASS_GLOB = amass_clip("ACCAD", "Male2Walking_c3d", "*.npz")


@requires_amass
@requires_smplx
def test_smplh_adapter_produces_smplx_frames():
    matches = sorted(glob.glob(AMASS_GLOB))
    if not matches:
        pytest.skip(f"no clips at {AMASS_GLOB}")
    f = matches[0]
    smplx_data, body_model, smplx_output, height = load_smplh_amass_file(f, BODY_MODELS)
    # adapter must expose SMPL-X-style keys consumed downstream
    assert smplx_data["pose_body"].shape[1] == 63
    assert smplx_data["root_orient"].shape[1] == 3
    assert "mocap_frame_rate" in smplx_data
    assert 1.0 < height < 2.5
    frames, fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30)
    assert len(frames) > 0
    # SMPL-X joint names used by IK configs must be present
    for j in ["pelvis", "left_hip", "left_knee", "left_foot", "spine3",
              "left_shoulder", "left_elbow", "left_wrist"]:
        assert j in frames[0], f"missing joint {j}"
    pos, quat = frames[0]["pelvis"]
    assert pos.shape == (3,) and quat.shape == (4,)
    assert abs(fps - 30) < 5


# ---- per-dataset framerate overrides (wrong AMASS metadata) ----

from retargeting.utils.smpl import true_frame_rate, FRAMERATE_OVERRIDES


def test_bmlhandball_claimed_rate_is_overridden_to_240():
    # AMASS stamps BMLhandball 120; the source paper says 240 Hz Vicon capture
    f = "/data/amass/BMLhandball/S01_Expert/Trial_upper_left_005_poses.npz"
    assert true_frame_rate(f, 120.0) == 240.0


def test_other_datasets_keep_their_claimed_rate():
    assert true_frame_rate("/data/amass/CMU/55/55_07_poses.npz", 120.0) == 120.0
    assert true_frame_rate("/data/amass/KIT/359/walking_run05_poses.npz", 100.0) == 100.0


def test_override_matches_the_path_segment_not_substrings():
    # a dataset merely containing the name must not match (segment-delimited)
    assert true_frame_rate("/x/motions/NotBMLhandballX/sub/clip_poses.npz", 120.0) == 120.0
    assert "BMLhandball" in FRAMERATE_OVERRIDES
