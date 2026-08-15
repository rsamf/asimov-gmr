# Third-party code, data, and models

This repository is MIT licensed (see `LICENSE`). That covers **the code in this
repository**. Running the pipeline end to end additionally requires data and
models that are **not distributed here** and that carry their own, more
restrictive terms. Read this before assuming a commercial use is permitted.

## Code

| Component | License | Notes |
|---|---|---|
| [GMR](https://github.com/YanjieZe/GMR) — the IK engine in `retargeting/` | MIT | Substantially upstream; notice reproduced in `NOTICE`. |
| `retargeting/torch_utils.py` | BSD-3-Clause | © 2018-2022 NVIDIA Corporation, from IsaacGym `poselib`. Header retained. |
| [mink](https://github.com/kevinzakka/mink), [MuJoCo](https://github.com/google-deepmind/mujoco) | Apache-2.0 | IK solver and physics. |
| numpy, scipy, flask, torch | BSD-3-Clause | |
| imageio, imageio-ffmpeg | BSD-2-Clause | Ships LGPL FFmpeg binaries at install time. |
| natsort, rich | MIT | |
| shadcn/ui, Radix UI, TanStack Table, Tailwind, lucide | MIT / ISC | Clip explorer frontend. |
| IBM Plex fonts | SIL OFL-1.1 | Bundled only in a built frontend, not in this repo. |
| **`smplx`** (pip package) | **Non-commercial research only** | Max Planck license. See below. |

## Data and models — you must obtain these yourself

### AMASS (source motion)

> **AMASS is licensed for non-commercial scientific research only, and may not
> be redistributed.**

Register and download at <https://amass.is.tue.mpg.de/>. Its license permits
creating derivative works but **not making them available to third parties**.
That is why this repository ships the *retargeting code and curation lists* —
clip names, thresholds, and metrics — but **no motion data**, and why the
retargeted dataset produced by this pipeline cannot be published either.

Each AMASS sub-dataset (CMU, KIT, ACCAD, BMLmovi, BMLhandball, MPI_HDM05,
Eyes_Japan, SFU, TotalCapture, EKUT, HumanEva, HUMAN4D, Transitions,
BioMotionLab_NTroje) additionally carries its own upstream citation
requirement; cite the collections you actually use.

### SMPL-X body models

> **SMPL-X models are gated and must not be redistributed.**

Register at <https://smpl-x.is.tue.mpg.de/>, accept the license, and download
`SMPLX_NEUTRAL.npz`, `SMPLX_MALE.npz`, `SMPLX_FEMALE.npz` into
`assets/body_models/smplx/` (gitignored). `scripts/fetch_smplx_models.py`
prints these instructions; it deliberately does **not** pull from unofficial
mirrors that bypass the registration gate.

The `smplx` Python package is likewise licensed for non-commercial scientific
research. Because the pipeline depends on it, **the MIT grant on this
repository does not make the end-to-end pipeline commercially usable.**

### asimov robot description

MuJoCo XML and meshes come from
<https://github.com/menloresearch/asimov-1> under that repository's terms. They
are not vendored here.

## Summary

- **This repository's code:** MIT — use it freely, keep the notices.
- **The pipeline as a whole:** constrained to non-commercial research by AMASS,
  SMPL-X, and `smplx`.
- **Anything this pipeline produces from AMASS:** derivative of AMASS. Keep it
  private unless you obtain written permission from Max Planck
  (`ps-license@tue.mpg.de`).
