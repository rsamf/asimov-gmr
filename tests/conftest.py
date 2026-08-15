"""Shared test fixtures and skip guards.

Most of this suite is pure logic and runs anywhere. The parts that exercise the
real IK need three things a fresh clone does not have — the AMASS corpus, the
SMPL-X body models, and the asimov robot description — so those tests SKIP
rather than fail when the data is absent.

Point the suite at your copies with:
    ASIMOV_AMASS_DIR=/path/to/AMASS
    ASIMOV_ROBOT_DIR=/path/to/asimov-1   (or a sibling clone of this repo)
"""
import os
import pathlib

import pytest

ROOT = pathlib.Path(__file__).parent.parent

AMASS_DIR = os.environ.get("ASIMOV_AMASS_DIR", "")
BODY_MODELS = str(ROOT / "assets" / "body_models")


def _robot_available():
    try:
        from retargeting.params import resolve_asimov_xml
        resolve_asimov_xml()
        return True
    except Exception:
        return False


def amass_clip(*parts):
    """Absolute path to a clip inside the configured AMASS corpus."""
    return os.path.join(AMASS_DIR, *parts)


requires_amass = pytest.mark.skipif(
    not os.path.isdir(AMASS_DIR),
    reason=f"AMASS corpus not found at {AMASS_DIR} (set ASIMOV_AMASS_DIR)")

requires_smplx = pytest.mark.skipif(
    not (ROOT / "assets" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz").exists(),
    reason="SMPL-X body models not installed (see scripts/fetch_smplx_models.py)")

requires_robot = pytest.mark.skipif(
    not _robot_available(),
    reason="asimov robot description not found (clone menloresearch/asimov-1 "
           "or set ASIMOV_ROBOT_DIR)")
