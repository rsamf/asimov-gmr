"""Unit tests for the limit-aware foot-tilt clamp (pure math, no models).

The clamp replaces the constant-alpha shrink: it must be the IDENTITY whenever the
human's foot tilt is already inside the ankle's range, and land exactly ON the
boundary when it is not -- while never touching heading (yaw).
"""
import pathlib
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import smplx_to_asimov as S  # noqa: E402

ROLL_MAX = np.radians(5.7 * 0.9)
PITCH_MAX = np.radians(20.1 * 0.9)


def _tilt_of(quat_wxyz):
    """(roll, pitch) of a quat after removing heading — the clamp's own decomposition."""
    R = Rot.from_quat(quat_wxyz, scalar_first=True)
    M = R.as_matrix()
    lvl = Rot.from_euler("z", np.arctan2(M[1, 0], M[0, 0]))
    r, p, _ = (lvl.inv() * R).as_euler("xyz")
    return r, p


def _quat(roll_deg, pitch_deg, yaw_deg=0.0):
    return (Rot.from_euler("z", yaw_deg, degrees=True)
            * Rot.from_euler("xyz", [roll_deg, pitch_deg, 0.0], degrees=True)
            ).as_quat(scalar_first=True)


def test_identity_when_already_reachable():
    q = _quat(3.0, 10.0)                      # inside both ankle ranges
    out = S._clamp_tilt_quat(q, ROLL_MAX, PITCH_MAX)
    r, p = _tilt_of(out)
    assert np.degrees(r) == pytest.approx(3.0, abs=1e-6)
    assert np.degrees(p) == pytest.approx(10.0, abs=1e-6)


def test_clamps_excess_roll_to_the_boundary():
    for demand in (8.0, 30.0, 60.0):
        r, _ = _tilt_of(S._clamp_tilt_quat(_quat(demand, 0.0), ROLL_MAX, PITCH_MAX))
        assert r == pytest.approx(ROLL_MAX, abs=1e-6), f"roll {demand} not clamped"
    r, _ = _tilt_of(S._clamp_tilt_quat(_quat(-45.0, 0.0), ROLL_MAX, PITCH_MAX))
    assert r == pytest.approx(-ROLL_MAX, abs=1e-6), "negative roll must clamp too"


def test_clamps_pitch_independently_of_roll():
    # roll reachable, pitch far out: roll must survive untouched
    r, p = _tilt_of(S._clamp_tilt_quat(_quat(4.0, 50.0), ROLL_MAX, PITCH_MAX))
    assert np.degrees(r) == pytest.approx(4.0, abs=1e-6)
    assert p == pytest.approx(PITCH_MAX, abs=1e-6)


def test_heading_is_preserved():
    for yaw in (-150.0, -30.0, 0.0, 95.0):
        out = S._clamp_tilt_quat(_quat(40.0, 40.0, yaw), ROLL_MAX, PITCH_MAX)
        M = Rot.from_quat(out, scalar_first=True).as_matrix()
        got = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
        assert ((got - yaw + 180) % 360) - 180 == pytest.approx(0.0, abs=1e-6)


def test_clamped_result_is_always_within_limits():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = _quat(*rng.uniform(-70, 70, size=2), yaw_deg=rng.uniform(-180, 180))
        r, p = _tilt_of(S._clamp_tilt_quat(q, ROLL_MAX, PITCH_MAX))
        assert abs(r) <= ROLL_MAX + 1e-9
        assert abs(p) <= PITCH_MAX + 1e-9


def test_margin_keeps_targets_off_the_hard_stop():
    # the clamp targets a fraction of the range, so a saturating demand lands
    # strictly inside the joint's true limit rather than pinned on it
    hard = np.radians(5.7)
    r, _ = _tilt_of(S._clamp_tilt_quat(_quat(90.0, 0.0), ROLL_MAX, PITCH_MAX))
    assert abs(r) < hard, "clamped roll must sit inside the hard stop"
    assert abs(r) == pytest.approx(0.9 * hard, rel=1e-6)
