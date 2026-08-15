# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

**Asimov GMR** retargets AMASS human motion onto the asimov humanoid and ships
the review UI for the result. `retargeting/` is the IK engine (derived from
upstream GMR); everything else is the asimov pipeline that drives it. Full
engineering write-up with measurements and rejected approaches:
**`docs/asimov_retargeting_report.md`**.

## Ground rules

- **Never commit data.** AMASS motion, SMPL-X models, retargeted pkls/npz, and
  review MP4s are all non-redistributable or huge. `.gitignore` covers them —
  don't add exceptions. See `THIRD_PARTY.md`.
- **Keep the notices.** ~90% of `retargeting/` is upstream GMR (MIT, Yanjie Ze)
  and `torch_utils.py` is NVIDIA BSD-3. `LICENSE` + `NOTICE` must stay accurate;
  never re-vendor CC BY-NC-ND code (the old LAFAN1 helpers were removed for this
  reason — `retargeting/utils/quat.py` replaced the one function used).
- **No machine-specific paths.** Everything resolves from a flag or an env var:
  `ASIMOV_AMASS_DIR`, `ASIMOV_ROBOT_DIR`/`ASIMOV_ROBOT_XML`, `ROBOTICS_ROOT`.
- **Frozen constants stay frozen.** Difficulty cutoffs, the glitch detector, fall
  thresholds and framerate overrides carry the measurement that set them. Change
  them deliberately, update the provenance comment, and expect labels to move.

## Environment

```bash
.venv/bin/python -m pytest tests/     # 82 pass / 5 skip without data
```

Tests skip cleanly when AMASS, SMPL-X or the robot description is absent (see
`tests/conftest.py`). torch is CPU-only here; all FK/IK runs on CPU.

## The one command

```bash
asimov-gmr run --amass <AMASS> --robot <asimov-1> --out <OUT> [--videos] [--verify]
```

`retargeting/pipeline.py` chains the stages below by invoking each script as a
subprocess — it is orchestration only, so every stage stays independently
runnable and debuggable:

```
retarget → metrics → difficulty → compile → fallen → merge → recompile → [videos]
```

| script | role |
|---|---|
| `scripts/smplx_to_asimov.py` | single-clip retarget + **the whole IK pipeline** (chord synthesis, decoupling, foot leveling, contact grounding, pinning) |
| `scripts/smplx_to_asimov_dataset.py` | batch driver (parallel, resumable, OOM-hardened) |
| `scripts/compile_training_dataset.py` | pkl → training npz; joins difficulty + split |
| `scripts/scan_fallen_refs.py` | drop refs that self-trigger the env's fall check |
| `scripts/make_test_split.py` | one-time frozen 60/60/60 split selection |
| `scripts/retarget_and_view.py` | interactive tuning viewer |
| `scripts/feasibility_scan.py` | joint-limit saturation diagnostics |
| `clip_explorer/compute_metrics.py` | float% + pelvis-relative residuals |
| `clip_explorer/compute_difficulty.py` | easy/medium/hard (`--calibrate` to re-freeze) |
| `clip_explorer/render_videos.py` | per-clip review MP4s |
| `clip_explorer/app.py` | the review UI (localhost:5001) |

## Architecture notes

- **IK**: mink/MuJoCo, solver `daqp`, damping 5e-1; **two sequential passes per
  frame** (`ik_match_table1` then `ik_match_table2`).
- **Config**: `configs/smplx_to_asimov.json` is THE config; importing
  `scripts/smplx_to_asimov.py` registers it for everything downstream. Field
  reference in `configs/README.md`.
- **Quaternions**: config and training npz use **wxyz**; saved pkls store
  `root_rot` as **xyzw**. Convert explicitly when crossing formats.
- **Clip identity**: `<DATASET>__<subject>__<clip>` everywhere (npz stem,
  manifest, metrics, split). `retargeting/utils/clip_names.py` owns the mapping;
  AMASS layout is derived from the path *shape*, not a hardcoded root.
- **Explorer columns are server-driven**: add a column in `app.py` and it
  appears; touch `frontend/src/lib/enrichment.ts` only for semantic rendering.
  Releases are **discovered** under `ROBOTICS_ROOT`, not enumerated.

## Hard-won decisions (don't re-litigate without reading the report)

- **Grounding**: exact per-geom lowest point (mesh vertices + collision-sphere
  `center−radius`) — NOT body origin (the ankle_roll origin sits 3.4 cm above the
  sole) and NOT rotated-AABB.
- **Chord synthesis** (`SPINE3_ABOUT_HIPS`): torso tilt targets the
  hip_center→spine3 chord; base rides it at radii calibrated at the most-upright
  frame, from clean scaled anatomy (task `pos_offsets` are link placement, not
  anatomy).
- **Partial foot leveling (α=0.4)**: a constant config `rot_offset` cannot cancel
  *variable* human tilt; full leveling kills kick articulation.
- **Weights**: foot pos-weight is the decisive fidelity lever; hip/knee position
  weights **overfit** (helped walk/kick, hurt run/punch across a 6-clip sweep).
- **Arms**: position-track elbow/wrist, orientation free; upper body decoupled
  from the base solve (`DECOUPLE_ARMS_FROM_BASE`).
- **Foot-lock: investigated and rejected** — stance drift is already ~1 cm; the
  scary 7–60 cm numbers were a contact-detection artifact.
- **Grounding bob is conservation-bounded** — within a stance, foot-lowest
  variation must land in body-z, foot clearance, or the solve. The candidate
  mechanisms are in `smplx_to_asimov.py` OFF with their measurements.
- **Excluded motion classes**: treadmill (feet skate ~49 cm/s by construction),
  prop/structure-dependent motion (chairs, benches, beams, ramps, stairs), and
  references that already trip the fall check (cartwheels, handstands).

## Reproducibility artifacts

`curation/` holds what makes a rerun match: `clips.txt` (exact source clips),
`rejects.json`, `test_split.json`, `expected.json` (the `--verify` target).
These are clip *names* and thresholds — never motion data.
