"""Height normalization: measured subject height + fixed-robot scale ratio.

Two root fixes under test (2026-07, the CMU/105 shrunken-figure diagnosis):
  1. measured_height(): subject height comes from T-posing the clip's own body
     model (shape/gender-correct), anchored so the zero-betas mean shape reads
     1.66 -- replacing the 1.66 + 0.1*betas[0] guess.
  2. GMR scales the human_scale_table by assumption/actual (NOT actual/
     assumption): the posed joints are already subject-true-size, so the old
     direction made target size ~ height^2 (a 1.43 m subject produced a 0.73 m
     target skeleton; 68% of the corpus ran >5% off nominal size).
"""
import os
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "scripts"))

BODY = HERE.parent / "assets" / "body_models" / "smplx"

pytestmark = pytest.mark.skipif(
    not (BODY / "SMPLX_NEUTRAL.npz").exists(),
    reason="needs SMPL-X body models",
)


def _model(gender="neutral"):
    import smplx
    return smplx.create(str(BODY.parent), "smplx", gender=gender,
                        use_pca=False, num_betas=16)


def test_zero_betas_reads_anchor():
    from retargeting.utils.smpl import measured_height
    h = measured_height(_model(), np.zeros(16))
    assert h == pytest.approx(1.66, abs=1e-6), "mean shape must read the anchor height"


def test_measured_height_monotone_in_beta0():
    from retargeting.utils.smpl import measured_height
    m = _model()
    hs = [measured_height(m, np.r_[b0, np.zeros(15)]) for b0 in (-2.0, 0.0, 2.0)]
    assert hs[0] < hs[1] < hs[2], f"height must grow with beta0: {hs}"
    # extremes stay anatomically plausible (the corpus spans ~1.4 - 2.0 m)
    assert 1.2 < hs[0] < 1.66 and 1.66 < hs[2] < 2.2


def test_scale_ratio_normalizes_short_subjects_up():
    """A short subject's table must be scaled UP (assumption/actual), so the
    already-small posed skeleton lands at robot-reference size."""
    import retargeting as g
    if not os.path.exists(str(g.ROBOT_XML_DICT["asimov"])):
        pytest.skip("asimov xml not on this machine")
    from retargeting import GeneralMotionRetargeting as GMR
    short = GMR(src_human="smplx", tgt_robot="asimov",
                actual_human_height=1.5, verbose=False)
    tall = GMR(src_human="smplx", tgt_robot="asimov",
               actual_human_height=1.8, verbose=False)
    ks = short.human_scale_table
    for k, v in tall.human_scale_table.items():
        assert ks[k] > v, f"{k}: short-subject scale {ks[k]} must exceed {v}"
    assert ks["left_foot"] == pytest.approx(v_ := tall.human_scale_table["left_foot"] * 1.8 / 1.5), \
        f"expected exact assumption/actual scaling, got {ks['left_foot']} vs {v_}"
