"""Check for the SMPL-X body models, and explain how to obtain them.

SMPL-X is distributed by the Max Planck Institute behind a registration and
license gate, and may NOT be redistributed. This script therefore does not
download anything: it verifies the files are in place and otherwise prints the
official instructions. Do not "fix" this by pointing it at an unofficial
mirror — routing around the gate violates the SMPL-X license.
"""
import pathlib
import sys

DEST = pathlib.Path(__file__).parent.parent / "assets" / "body_models" / "smplx"
FILES = ["SMPLX_NEUTRAL.npz", "SMPLX_MALE.npz", "SMPLX_FEMALE.npz"]
URL = "https://smpl-x.is.tue.mpg.de/"

INSTRUCTIONS = f"""\
SMPL-X body models are required but are not distributed with this repository.

  1. Create an account at {URL} and accept the license.
  2. Download "SMPL-X v1.1 (NPZ+PKL, 830 MB)" from the Downloads page.
  3. Extract it and copy these files into
       {DEST}
     {", ".join(FILES)}

The pipeline needs the .npz variants (it loads them with num_betas=16 and
use_pca=False). The directory is gitignored — never commit these files.
"""


def missing():
    return [f for f in FILES if not (DEST / f).exists()]


def main():
    absent = missing()
    if absent:
        print(INSTRUCTIONS)
        print(f"missing {len(absent)}/{len(FILES)}: {', '.join(absent)}")
        return 1
    for f in FILES:
        p = DEST / f
        print(f"ok: {p} ({p.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
