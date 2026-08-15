"""Unit tests for stance foot-target pinning (_FootPinner), synthetic data only.

The pin must remove sub-threshold mocap creep while a foot is planted, but must
NOT fight genuine grounded movement: fast motion breaks the contact mask and
releases; slow deliberate slides release via the max-drift escape; and it is
position-only by design (orientation is never touched).
"""
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import smplx_to_asimov as S  # noqa: E402


def _frames(left_xyz, right_xyz):
    return [{"left_foot": (np.array(l, float), None),
             "right_foot": (np.array(r, float), None)}
            for l, r in zip(left_xyz, right_xyz)]


def _still(n, xyz=(0.0, 0.0, 0.02)):
    return [list(xyz)] * n


def test_pin_holds_xy_against_subthreshold_creep():
    n = 30
    # left foot creeps 3 cm over 30 frames (0.1 cm/frame, well under the
    # velocity threshold) while staying on the floor -> mocap creep, pin it
    left = [[0.001 * i, 0.0, 0.02] for i in range(n)]
    fr = _frames(left, _still(n))
    p = S._FootPinner(fr)
    outs = [p.pos("left_foot", np.array(l)) for l in left]
    # anchored at frame 0; all subsequent outputs hold the anchor XY
    assert np.allclose(outs[0][:2], [0.0, 0.0])
    for o in outs[1:]:
        assert np.allclose(o[:2], outs[0][:2]), "pinned foot XY must not creep"


def test_z_pinned_with_flag_live_without():
    """Z-pinning is the bob fix: the planted foot's z TARGET wanders 1-3.5 cm
    within a stance (root-relative scaling leaks the human root's bobbing into
    the foot target), the IK tracks it, and per-frame grounding re-datums the
    whole body. Freezing z at the stance anchor removes that at the source.
    (Superseded design: z used to stay live -- that let the wander through.)"""
    n = 10
    left = [[0.0, 0.0, 0.02 + 0.001 * i] for i in range(n)]   # z creeps up in stance
    fr = _frames(left, _still(n))
    old = S.PIN_FOOT_Z
    try:
        S.PIN_FOOT_Z = True
        p = S._FootPinner(fr)
        outs = [p.pos("left_foot", np.array(l)) for l in left]
        assert outs[-1][2] <= outs[0][2] + 1e-12, "flag on -> Z holds the stance anchor"
        S.PIN_FOOT_Z = False               # shipped default (v5 behavior)
        p = S._FootPinner(fr)
        outs = [p.pos("left_foot", np.array(l)) for l in left]
        assert outs[-1][2] == pytest.approx(left[-1][2]), "flag off -> Z tracks live"
    finally:
        S.PIN_FOOT_Z = old


def test_fast_motion_releases_and_blends_to_live():
    # planted 10 frames, then the foot moves fast (breaks the velocity mask)
    still = [[0.0, 0.0, 0.02]] * 10
    moving = [[0.05 * (i + 1), 0.0, 0.10] for i in range(10)]   # 5 cm/frame, lifted
    left = still + moving
    fr = _frames(left, _still(len(left)))
    p = S._FootPinner(fr)
    outs = [p.pos("left_foot", np.array(l)) for l in left]
    # by the end of the blend the output must equal the live (moving) target
    assert np.allclose(outs[-1], left[-1]), "after release the live target wins"
    # and the frame right after release must NOT snap all the way instantly
    first_after = outs[10]
    assert np.linalg.norm(first_after[:2] - np.array(left[10][:2])) > 1e-6, \
        "release should blend, not pop"


def test_slow_deliberate_slide_releases_via_max_drift():
    # foot stays "planted" by the mask (slow + on floor) but slides far: after
    # the live target exceeds max_drift from the anchor, the pin must let go
    n = 60
    left = [[0.005 * i, 0.0, 0.02] for i in range(n)]   # 0.5 cm/frame, 30 cm total
    fr = _frames(left, _still(n))
    p = S._FootPinner(fr)   # PIN_MAX_DRIFT = 0.10
    outs = [p.pos("left_foot", np.array(l)) for l in left]
    # far past the drift threshold + blend, output must follow the live slide
    end_err = np.linalg.norm(outs[-1][:2] - np.array(left[-1][:2]))
    assert end_err < 0.06, f"pin fought a deliberate grounded slide (err {end_err:.3f} m)"
    # but the early sub-drift portion was pinned
    assert np.allclose(outs[5][:2], outs[1][:2])


def test_per_foot_masks_are_independent():
    n = 12
    left = _still(n)                                     # planted
    right = [[0.05 * i, 0.0, 0.30] for i in range(n)]    # swinging high + fast
    masks = S._per_foot_contact(_frames(left, right))
    assert masks["left_foot"].all()
    assert not masks["right_foot"].any()
