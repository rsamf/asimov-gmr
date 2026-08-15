import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from retargeting.utils.quat import quat_mul


def _rand_unit(rng, n):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def test_matches_scipy_composition():
    """a ⊗ b must equal composing the two rotations (scipy, scalar-first)."""
    rng = np.random.default_rng(0)
    a, b = _rand_unit(rng, 64), _rand_unit(rng, 64)
    got = quat_mul(a, b)
    want = (R.from_quat(a, scalar_first=True) * R.from_quat(b, scalar_first=True)) \
        .as_quat(scalar_first=True)
    # quaternions double-cover SO(3): q and -q are the same rotation
    agree = np.isclose(got, want, atol=1e-12) | np.isclose(got, -want, atol=1e-12)
    assert agree.all()


def test_agrees_with_the_implementation_it_replaced():
    """The removed vendored helper summed the same products in a different
    order, so agreement is to floating-point precision, not bit-for-bit. This
    bounds the drift for releases built before the replacement."""
    rng = np.random.default_rng(1)
    x, y = _rand_unit(rng, 512), _rand_unit(rng, 512)
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]
    legacy = np.concatenate([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], axis=-1)
    assert np.abs(quat_mul(x, y) - legacy).max() < 1e-15


def test_identity_and_inverse():
    rng = np.random.default_rng(2)
    q = _rand_unit(rng, 16)
    ident = np.tile([1.0, 0.0, 0.0, 0.0], (16, 1))
    assert np.allclose(quat_mul(q, ident), q)
    assert np.allclose(quat_mul(ident, q), q)
    conj = q * np.array([1.0, -1.0, -1.0, -1.0])
    assert np.allclose(quat_mul(q, conj), ident, atol=1e-12)


def test_broadcasts_and_accepts_single_quaternions():
    rng = np.random.default_rng(3)
    one, many = _rand_unit(rng, 1)[0], _rand_unit(rng, 8)
    assert quat_mul(one, one).shape == (4,)
    assert quat_mul(one, many).shape == (8, 4)          # the smpl.py call shape
    assert np.allclose(quat_mul(one, many)[3], quat_mul(one, many[3]))


def test_order_matters():
    rng = np.random.default_rng(4)
    a, b = _rand_unit(rng, 4), _rand_unit(rng, 4)
    assert not np.allclose(quat_mul(a, b), quat_mul(b, a))
