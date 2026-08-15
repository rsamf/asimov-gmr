---
title: Asimov Clip Explorer
emoji: 🤖
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Asimov Clip Explorer

Review UI for AMASS motion retargeted onto the **asimov** humanoid: per-clip IK
quality, dynamics, difficulty labels, train/test split, and training feedback,
with the retargeted video for every clip.

Built from [rsamf/asimov-gmr](https://github.com/rsamf/asimov-gmr) and deployed
by CI on every push to `main`.

## Data

This Space serves **precomputed metadata and videos** from a dataset repo — it
does not retarget anything. Configure:

| variable | | |
|---|---|---|
| `HF_DATASET_REPO` | variable | `<owner>/<name>` of the dataset repo holding a release |
| `HF_TOKEN` | **secret** | read token; required because that repo is private |

The dataset repo is private by necessity: the clips derive from
[AMASS](https://amass.is.tue.mpg.de/), whose license forbids redistribution.
The pipeline that produces them is open source — run it on your own AMASS copy:

```bash
asimov-gmr run --amass <AMASS> --robot <asimov-1> --out ./out
```

With no dataset configured the Space boots and reports that no releases were
found, which is the expected state for a fork.
