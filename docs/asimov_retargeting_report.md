# Asimov Motion Retargeting — Engineering Report

**Goal:** Produce the most accurate possible humanoid motion dataset for the **asimov** robot by retargeting human (AMASS SMPL+H) motion through the GMR pipeline, delivering ≥3.5 h of clean, training-ready motion.

**Output:** one `.pkl` per clip + `manifest.json` under the pipeline's `--out` directory.

---

## 1. Summary of outcome

A complete retargeting pipeline for asimov was built from scratch (the robot was not previously supported in GMR), rigorously tuned for foot-placement quality, and run over a curated, dataset-balanced selection of AMASS motion with treadmill clips excluded.

Final per-clip quality (measured across walk / run / treadmill / kick / punch / female-walk):

| Metric | Baseline (G1-derived config) | Final tuned config |
|---|---|---|
| Foot-target fidelity (achieved vs. human foot) | 2.5 cm | **0.95 cm** |
| Locomotion foot hover | 4.7 cm | **2.3 cm** |
| Foot articulation (kick foot tilt) | 24° | 16° (preserved) |
| Stance-phase foot drift (skate) | ~1 cm (already low) | ~1 cm |
| **Arm end-effector (wrist) position error** | 24.6 cm | **1.8 cm** |
| **Elbow joint saturation** | 55% of frames | **9%** |

Batch results: **2064 clips, 0 failures, 4.50 h, 485,486 frames, 271 MB** — passes full QC (no NaNs, no joint-limit violations).

---

## 2. Starting point and key constraints discovered

- **asimov was unsupported.** A new IK config, params registration, and a SMPL+H input adapter all had to be built.
- **Source data is AMASS SMPL+H**, not SMPL-X (`poses (N,156)`, `dmpls`, `trans`, `betas`, `gender`, `mocap_framerate`). GMR's loader expects SMPL-X keys, so an adapter (`load_smplh_amass_file`) slices `poses[:, :3]`→root_orient, `poses[:, 3:66]`→body_pose and feeds the shared SMPL-X body model.
- **No SMPL body models on disk.** SMPL-X neutral/male/female are obtained from the official gated source (<https://smpl-x.is.tue.mpg.de/>, registration + license acceptance required) into `assets/body_models/smplx/`; `scripts/fetch_smplx_models.py` verifies they are present. They are never redistributed with this repository.
- **No conda / GMR not installed.** Set up a pyenv-3.10 venv with `pip install -e .` + deps (mink, mujoco, smplx, torch-CPU). No GPU on this machine (`torch.cuda.is_available()` is False); all FK uses CPU.
- **asimov structure:** 25-DOF humanoid, near-twin of Unitree G1. Free base `floating_base` (nq=32, nv=31). Foot = `ankle_roll_link` (no toe body). All link frames world-aligned (`quat=1 0 0 0`); pelvis at 0.63 m (G1: 0.793 m — asimov is smaller).

## 3. Robot onboarding

- `smplx_to_asimov.json` IK config generated from the G1 config by remapping link names (`pelvis`→`pelvis_link`, foot→`ankle_roll_link`, `torso_link`→`waist_yaw_link`); G1 rotation offsets transferred because both robots share the world-aligned frame convention.
- Registered `asimov` in `params.py` (`ROBOT_XML_DICT`, `IK_CONFIG_DICT["smplx"]`, `ROBOT_BASE_DICT`="pelvis_link", `VIEWER_CAM_DISTANCE_DICT`); XML referenced in place (meshdir resolves relative to the XML).
- Headless renderer (`clip_explorer/render_videos.py`) built on `mujoco.Renderer` + EGL (the stock `RobotMotionViewer` needs X11), with a base-tracking camera so translating motions stay framed. Verified the wxyz↔xyzw quaternion round-trip composes to identity.

## 4. Grounding (foot-floor contact)

Three iterations, each fixing the previous:
1. **Body-origin grounding** (GMR default) sank the foot ~3.7 cm — asimov's `ankle_roll_link` origin sits 3.4 cm above the sole.
2. **Rotated-AABB geometry grounding** lifted the robot ~3.4 mm (an AABB corner sits below the true rotated mesh surface).
3. **Final: exact per-geom lowest point** (mesh vertices for the visual foot, `center−radius` for collision spheres) over the whole clip → the visible sole rests at z=0, nothing penetrates.

## 5. Foot-placement tuning (the core effort)

The dominant quality issue was the planted foot appearing to **hover** ~5 cm. Root cause (established by measurement, not guessing): GMR grounds a whole clip by a single offset, resting the *single lowest instant* on the floor. The foot's lowest point swings as the foot tilts during the gait, so most frames float. The achieved *ankle position* was already consistent to ±1.3 cm — the hover was driven by foot **orientation**, not position.

Approaches explored and their verdicts:

- **Per-frame grounding** — plants every frame, but flattens true flight phases (jumps); rejected (user preferred the smooth single-offset look).
- **Active foot leveling (force flat)** — drove hover to ~1 cm and pitch to ~0°, but killed foot articulation and made the kick look wrong; rejected (tilt is important).
- **Partial foot leveling (chosen):** scale the human foot tilt toward level by a factor α (keep heading, shrink pitch/roll). A continuous dial between full articulation and flat. Implemented as a runtime target override in `_retarget_frame` (a constant config offset cannot do this — it must cancel the *variable* human tilt).
- **Hip/knee position weights** — initially looked good on 3 clips, but a rigorous 6-clip sweep showed hip weight **overfits** (helped walk/kick, hurt run/punch). Dropped to 0. Knee weight also dropped to 0 (hurts foot tracking and balance, per the user's hypothesis — confirmed).
- **Foot position weight 100→200** — the decisive fidelity lever, found in the rigorous sweep: foot-target residual fell from ~3 cm to **0.95 cm** with no hover cost.
- **Foot rotation weight 40→20** — *increased* natural articulation (16° vs 13°) while staying smooth.

**Rigorous search:** two automated sweeps, **156 configurations × 6 diverse clips**, scoring locomotion hover + dynamic hover + foot-target fidelity + articulation + smoothness. Winner: **hip 0, knee 0, foot-pos 200, foot-rot 20, tilt α=0.4.**

## 5b. Arm overhaul (position-tracking)

The arm mapping was inherited verbatim from the G1 config and tracked shoulder/
elbow/wrist **orientation** (`pos_w=0, rot_w=10`). Measurement exposed two faults:
the elbow was driven to ~16° **hyperextension** (clipped straight in 50%+ of
frames), and the wrist end-effector landed **~25 cm** from the human wrist
(orientation tracking never constrains position; with different limb proportions
the wrist drifts).

Fix: switch the arm to **position-tracking** — target the human wrist position
(`pos_w=100`) plus a light elbow position pull (`pos_w=30`, to resolve the
elbow-circle redundancy toward the human's arm plane), orientation reduced to a
floor. This is consistent with how the legs already work (feet are position-
tracked). Compared across punch / kick / walk / BMLmovi:

| Arm mode | elbow saturation | wrist pos error | elbow pos error |
|---|---|---|---|
| orientation (old G1) | 55% | 24.6 cm | 19.8 cm |
| wrist-pos only | 15% | 1.4 cm | 7.4 cm |
| **wrist-pos + elbow-pos (chosen)** | **9%** | 1.8 cm | 6.7 cm |
| wrist-pos + elbow-orient | 69% | 3.9 cm | 7.5 cm |

End-effector error dropped **14×** (24.6→1.8 cm) and the hyperextension is gone
(the elbow now bends naturally to reach the wrist). Verified visually: bent-elbow
boxing guard on the punch, natural arm swing on the walk, and hands correctly
reaching the floor on a BMLmovi push-up (which orientation tracking could not do).
The residual ~6.7 cm elbow error is the irreducible upper-arm segment-length
mismatch (asimov upper-arm 9.6 cm vs scaled-human 16.8 cm — hardware).

## 6. Foot-skating investigation (and why it was *not* needed)

I suspected stance-foot skating would hurt RL training and prototyped a two-pass foot-lock. Initial metrics looked alarming (7–60 cm of stance drift) and motivated building it. On scrutiny, those numbers were a **measurement artifact**: the contact detector used a height threshold (3 cm), but since feet hover only ~2.3 cm, *swing* feet also dipped under it, merging multiple steps into one bogus "stance phase."

Re-running contact detection from the **human foot's horizontal speed** (which cleanly separates stance from swing) showed the *actual* stance drift is already **~1 cm** (walk 1.1, run 0.8, kick 0.7, fwalk 2.1). Foot-lock improved this only marginally (→~0.8 cm) at 2× compute and artifact risk. **Decision: skip foot-lock** — the tuned config already produces clean contacts; the right fix was the metric, not the motion. (Documented as a deliberate negative result.)

## 7. Curation

- **Treadmill clips excluded** (412 files): walk-in-place captures produce feet that slide ~49 cm/s relative to the ground — unphysical sliding-contact data that degrades RL.
- Excluded non-motion `shape.npz` (96) and hard-to-retarget motions (crawl / lying / stairs).
- **Balanced across datasets** (round-robin: ACCAD, Transitions, BMLhandball, BMLmovi, BioMotionLab) for motion variety rather than letting the largest dataset dominate.

## 8. Pipeline / reproduction

```bash
# env (pyenv 3.10 venv at .venv) already set up; models in assets/body_models/smplx/
.venv/bin/python scripts/smplx_to_asimov_dataset.py --target_hours 4.5 --workers 14
# single clip + video:
.venv/bin/python clip_explorer/render_videos.py <dataset_dir> <video_dir>
```

Output `.pkl` schema per clip: `{fps, root_pos (N,3), root_rot (N,4 xyzw), dof_pos (N,25), local_body_pos, link_body_list}`.

## 9. Known limitations (measured across the 485k-frame dataset)

Ranked by % of frames each joint sits within ~2° of a limit (i.e. where the robot
physically cannot match the human):

| Restriction | Frames at limit | Type | Fixable? |
|---|---|---|---|
| **Ankle roll** (±5.7° range) | **54–56%** | hardware | No — caps lateral foot leveling & lateral balance |
| ~~Elbow hyperextension~~ | ~~50–54%~~ → **9%** | calibration | **FIXED** (arm position-tracking, §5b) |
| Shoulder roll | 11–16% | mixed | improved by arm position-tracking |
| Ankle pitch (±20°) | 10–11% | hardware | No — limits toe-off / heel-strike |
| Hip yaw (±45°) | 5–8% | hardware | No — limits crossover/turn stances |
| Knee (max flexion 86°) | ~5% | hardware | No — limits deep squat/kneel/high-knee |

Other restrictions:
- **Rigid torso:** asimov has a *single* `waist_yaw` DOF — **no spine pitch or roll**. Bending/leaning/bowing motions cannot be reproduced faithfully; the torso stays upright and the hips compensate. Hardware.
- **Arms — FIXED via position-tracking (§5b).** Remaining arm limits are now hardware/proportion: the residual ~6.7 cm elbow position error (asimov's upper arm is physically shorter than the human's, so the elbow can't sit where the human's does), and the **yaw-only wrist** (hand orientation can't match the human's 3-DOF wrist regardless of tracking mode).
- **Hand-tip / end-effector reach:** the wrist is now tracked to ~1.8 cm, but the human hand extends ~9 cm beyond the wrist and asimov has no hand link. For **manipulation/contact** tasks, adding a hand link + targeting the human hand point would close this ~9 cm gap (a robot-model change, recommended only if manipulation is needed). For whole-body/locomotion the wrist end-effector is sufficient.
- **Head untracked** (neck fixed at 0) and **no fingers**: head orientation could be added cheaply via a neck→head mapping (data is present in SMPL+H, no SMPL-X needed); fingers would require an actuated hand. Both optional, task-dependent.
- **Residual hover ~2.3 cm:** floor for single-offset grounding while preserving foot tilt (per-frame grounding rejected to keep flight-phase fidelity).
- **SMPL+H vs SMPL-X:** we feed SMPL+H betas through a SMPL-X *body* model. The 22 body joints are shared between the two formats, so body *pose* transfers exactly; only *shape* (limb proportions) is mildly inconsistent. Measured shape sensitivity is ~1–2 cm of joint-position / ~1 cm of limb-length variation, so the format-mismatch error is sub-2 cm. asimov has no hands/face/fingers, so SMPL-X's extra DOFs are unusable. **Verdict:** switching to native SMPL-X data is at most a ~1 cm target-accuracy gain — not worth it unless chasing maximal fidelity.
- **Kinematic only:** no dynamic-feasibility check (torque/balance); standard for RL reference data (the policy learns dynamics), but the motions are not guaranteed physically executable as-is.

## 12. Joint-limit saturation: cause and validated hardware fix

The retargeted data sits near several joint limits a large fraction of the time. Investigation (data-selection, foot roll-leveling, nullspace posture task, arm-weight sweeps) showed this is **not** fixable in software — it is **range-bound hardware behavior**, pervasive across all clips (only 0.5 h of 4.5 h has <50% ankle-roll saturation, so curation cannot avoid it). The dominant cases:
- **ankle_roll (±5.7°):** the ankle maxes out trying to level the foot against the leg's lateral tilt — the retargeting actually demands up to ~20° of roll and is being **clipped** at ~6°.
- **shoulder_roll (one-sided [−90°,0] / [0,90°]):** roll=0 **is** the joint limit and also the arms-at-side posture, so any arms-down motion sits at the boundary by design.

These are **faithful** (every frame is limit-valid; the real asimov has the same limits, so a reference at the limit is what the robot can actually do) and the **motion still looks natural** — but "frames at limit" is high and intrinsic.

**Validated sanity check (kinematic — retargeting has no physics/self-collision):** widening the ranges in a test model and re-retargeting the same clips:

| joint | range change | saturation (orig → wide) | max angle used (orig → wide) |
|---|---|---|---|
| ankle_roll | ±5.7° → ±20° | 58% → **5.5%** | clipped at 6° → **20°** |
| ankle_pitch | ±20° → ±40° | 14.5% → **1.5%** | |
| shoulder_roll | one-sided → two-sided (±45° past 0) | 82% → **0%** | 42° → 43° (unchanged) |

The widened model produces natural walk/kick motion (no contortion; root/dof within normal ranges). This **confirms the saturation is purely range-bound**, and quantifies the benefit *if* the hardware allows these ranges.

**Recommendation:** the current narrow-model dataset is the correct, *feasible* reference for the **current** asimov (commanding beyond ±5.7° ankle roll would be unexecutable on the real robot, so the clamped data is right). But if the hardware/URDF can be revised, the highest-value changes — in order — are: **two-sided shoulder-roll**, **wider ankle-roll (→~±20°)**, and **wider ankle-pitch (→~±40°)**. Regenerate with the updated model and saturation drops to single digits.

## 10. Commits (branch `asimov-retargeting`)

```
26361f8 register asimov robot with G1-derived SMPL-X IK config
12b91a3 single-clip asimov retarget + headless MP4 renderer
51c00df base-tracking camera to asimov renderer
350fdb0 ground asimov retarget on foot geometry, not body origin
d4492e8 ground on exact full-geometry lowest point; lower QA camera
c4a79d3 tune asimov IK: partial foot-leveling + rigorous weight search
<this>  batch dataset script + report + 4.5h retargeted dataset
```

## 11. Batch results & verification

**Run:** `--target_hours 4.5 --workers 14 --seed 0`, ~40 min on 16-core CPU.

| Dataset | Hours | Notes |
|---|---|---|
| ACCAD | 0.42 | locomotion, martial arts (all usable clips) |
| Transitions_mocap | 0.23 | varied transitions (all usable clips) |
| BMLhandball | 1.50 | sports / handball motions |
| BMLmovi | 0.89 | everyday + exercise motions |
| BioMotionLab_NTroje | 1.46 | walking, running, lifting (treadmill excluded) |
| **Total** | **4.50** | 2064 clips, 485,486 frames |

**Programmatic QC (all 2064 pkls):** 0 NaN/inf, 0 malformed, 0 quaternion-norm errors, **0 joint-limit violations** (every frame within asimov's joint ranges — enforced by mink's ConfigurationLimit). Joint usage spans the full ranges → good training coverage. Neck joints fixed at 0 (head untracked, as in the source G1 config).

**Visual verification:** clips rendered and confirmed natural across walk, run, kick, punch (ACCAD), handball (sport), BMLmovi (everyday/exercise), and BioMotionLab lifting — feet planted on the ground, articulation preserved, no inverted/penetrating limbs.

**Per-clip `.pkl` schema:** `{fps, root_pos (N,3), root_rot (N,4 xyzw), dof_pos (N,25), local_body_pos (N,nbody,3), link_body_list}`. Root height grounded so the foot sole rests at z=0; first-frame XY centred at origin.

**Reproduce / extend:** rerun `scripts/smplx_to_asimov_dataset.py` with a larger `--target_hours` (up to ~11.5 h available after exclusions) to grow the dataset; the curation, balancing, and tuning are identical.
