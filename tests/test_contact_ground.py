"""Unit tests for contact-mask grounding (PBHC-style), fast and body-model-free.

`_human_contact_mask` is pure (synthetic frames). `_contact_ground_offsets` needs
the asimov XML (registered in ROBOT_XML_DICT) but no body models or source clips.
"""
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "scripts"))

from conftest import requires_robot  # noqa: E402
import smplx_to_asimov as S  # noqa: E402
from retargeting import ROBOT_XML_DICT  # noqa: E402

pytestmark = requires_robot


class _LazyXml:
    """Defer ROBOT_XML_DICT lookup to call time.

    Resolving at import would raise during collection on a machine without the
    robot description, turning a skip into a hard error.
    """

    def __str__(self):
        return str(ROBOT_XML_DICT["asimov"])


XML = _LazyXml()


def _frames(left, right):
    """Build minimal frame dicts carrying only the two foot positions."""
    return [{"left_foot": (left[i], None), "right_foot": (right[i], None)}
            for i in range(len(left))]


def test_contact_mask_flags_planted_not_swinging():
    n = 10
    # left foot planted on the floor the whole time (stationary, low)
    left = np.tile([0.0, 0.0, 0.02], (n, 1))
    # right foot swings high and moves fast every frame
    right = np.stack([[0.3 * i, 0.0, 0.4] for i in range(n)])
    mask = S._human_contact_mask(_frames(left, right))
    assert mask.all(), "a permanently planted foot must read as contact every frame"


def test_contact_mask_all_false_when_both_feet_airborne_and_moving():
    n = 8
    left = np.stack([[0.2 * i, 0.0, 0.5 + 0.1 * i] for i in range(n)])
    right = np.stack([[0.2 * i + 0.1, 0.0, 0.5 + 0.1 * i] for i in range(n)])
    mask = S._human_contact_mask(_frames(left, right))
    assert not mask.any(), "both feet fast and high (flight) must read as no contact"


def _neutral_qpos(zs):
    """Neutral standing pose (all joints 0, upright) at each root height in `zs`."""
    q = np.zeros((len(zs), 32))
    q[:, 3] = 1.0            # wxyz identity quat
    q[:, 2] = zs
    return q


def _lowest_after(grounded):
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(XML)); d = mujoco.MjData(m)
    PLANE = mujoco.mjtGeom.mjGEOM_PLANE
    gids = [g for g in range(m.ngeom)
            if m.geom_bodyid[g] != 0 and m.geom_type[g] != PLANE]
    cor = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
    out = np.empty(len(grounded))
    for i in range(len(grounded)):
        d.qpos[:] = grounded[i]; mujoco.mj_forward(m, d)
        out[i] = min(S._geom_lowest_z(m, d, g, cor) for g in gids)
    return out


def test_offsets_ground_stance_to_the_floor():
    # a realistic gently-bobbing stance (~2 cm), the regime EMA is meant for
    zs = np.array([0.80, 0.82, 0.80, 0.78, 0.80, 0.81, 0.79])
    q = _neutral_qpos(zs)
    contact = np.ones(len(zs), dtype=bool)
    grounded = q.copy(); grounded[:, 2] -= S._contact_ground_offsets(XML, q, contact)
    low = _lowest_after(grounded)
    # EMA leaves a small lag, but stance sits ~on the floor, not tens of cm up
    assert np.mean(np.abs(low)) < 0.02, f"stance mean float {np.mean(np.abs(low))*100:.1f} cm"
    assert np.max(np.abs(low)) < 0.05, f"stance max float {np.max(np.abs(low))*100:.1f} cm"


def test_no_frame_penetrates_the_floor():
    # a stance that dips fast enough for the EMA to lag into penetration without
    # the clamp; every frame must still end up at or above GROUND_CLEARANCE.
    zs = np.array([0.85, 0.80, 0.70, 0.62, 0.70, 0.80, 0.85])
    q = _neutral_qpos(zs)
    contact = np.ones(len(zs), dtype=bool)
    grounded = q.copy(); grounded[:, 2] -= S._contact_ground_offsets(XML, q, contact)
    low = _lowest_after(grounded)
    # RSI must never start underground: lowest geom >= clearance on every frame
    assert low.min() >= S.GROUND_CLEARANCE - 1e-6, \
        f"frame penetrates floor: min lowest {low.min()*100:.2f} cm < {S.GROUND_CLEARANCE*100:.2f} cm"
    assert S.GROUND_CLEARANCE > 0, "clearance must be strictly positive (> 0, not >= 0)"


def test_offsets_hold_through_flight_so_jumps_are_not_glued():
    zs = np.array([0.80, 1.30, 1.30, 0.80])      # up in the air on the middle frames
    q = _neutral_qpos(zs)
    contact = np.array([True, False, False, True])
    off = S._contact_ground_offsets(XML, q, contact)
    # the offset is nearly constant (held across flight), so the airborne frames
    # keep their extra height rather than being dropped to the floor
    assert off.std() < 0.03, f"offset varied too much across flight: std {off.std():.3f}"
    airborne = (q[:, 2] - off)[1:3]
    assert airborne.min() > (q[:, 2] - off)[[0, 3]].max() + 0.2, \
        "flight frames should stay well above the grounded stance frames"
