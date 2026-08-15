"""Quaternion helpers, scalar-first (wxyz) — the convention the SMPL-X adapter
and the IK config both use.

scipy is available and used everywhere else, but it normalizes on construction;
these operate on the raw components so results are bit-stable for already-unit
inputs and broadcast over leading dimensions.
"""
import numpy as np


def quat_mul(a, b):
    """Hamilton product a ⊗ b for wxyz quaternions.

    Composing rotations: applying `a ⊗ b` rotates by b first, then a.
    Broadcasts over leading axes; last axis must be 4.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)
