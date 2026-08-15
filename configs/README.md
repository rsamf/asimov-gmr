# GMR IK config reference

How the `smplx_to_<robot>.json` IK configs work, field by field. Verified against
`retargeting/motion_retarget.py` (the loader). Use this when
hand-tuning a config (e.g. `smplx_to_asimov_tune.json` with
`scripts/retarget_and_view.py`).

## What a config does

Each frame of human motion is a dict of `human_body → (position, orientation)`. The
retargeter (built on **mink**/MuJoCo IK) places the robot's **floating base** at the
human root, then solves joint angles so that a set of robot links track chosen human
bodies as weighted **frame targets** (position + orientation). The config defines that
mapping.

## Top-level fields

| field | meaning |
|---|---|
| `robot_root_name` | The robot's floating base body (e.g. `pelvis_link`). Its world pose is set directly from the human root; IK solves everything else relative to it. |
| `human_root_name` | The human body that drives the root (e.g. `pelvis`). |
| `ground_height` | Assumed floor height of the human data. Subtracted from every `pos_offset`, and used for height alignment. |
| `human_height_assumption` | Reference human height (m) the config was authored for. At load, `ratio = actual_human_height / human_height_assumption` and **every `human_scale_table` value is multiplied by `ratio`** — so the fit auto-adjusts per subject. |
| `use_ik_match_table1` / `use_ik_match_table2` | Enable/disable each of the two IK passes. |

## `human_scale_table`

Per-human-body scale applied to that body's **offset from the root**:

```
scaled_pos = (human_pos − root_pos) × scale        # for each body
scaled_root = root_pos × scale[human_root]          # root scales overall height/travel
```

It fits the human skeleton onto the robot's proportions (a humanoid smaller than the
`human_height_assumption` needs values `< 1.0` to pull limbs inward). Remember the
value you write is further multiplied by the height ratio, so **effective scale ≈
`written × (subject_height / human_height_assumption)`**.

Per-body entries (`left_knee`, `left_shoulder`, …) let you match each limb's length
independently. Overlay the human skeleton (`retarget_and_view.py --overlay`) to tune
these visually.

## The two match tables

IK runs in **two sequential passes per frame**: `ik_match_table1` then
`ik_match_table2`, each listing the same robot links with (usually different) weights.
It's a coarse-then-refine scheme — e.g. table1 may pin the feet with `pos=100, rot=10`
while table2 re-weights to `pos=50, rot=50` to lock foot orientation during
refinement.

## Each entry — the 5-tuple

```json
"robot_link": [ human_body, pos_weight, rot_weight, pos_offset[x,y,z], rot_offset[w,x,y,z] ]
```

| # | field | what it does |
|---|---|---|
| 1 | **human_body** | Which human body this robot link tracks. `"left_ankle_roll_link": ["left_foot", …]` → the robot foot link follows the human foot. |
| 2 | **pos_weight** | mink `position_cost`: how hard IK pulls this link's **origin** to the target position. `0` = don't track position. Relative magnitudes across links matter (one weighted least-squares solve). |
| 3 | **rot_weight** | mink `orientation_cost`: how hard IK matches this link's **orientation**. `0` = ignore orientation. |
| 4 | **pos_offset [x,y,z]** | Translation added to the target, **in the link's own local frame** (after `rot_offset`), meters. Example: the asimov feet use `[-0.15, ±0.02, 0]` to shift the ankle target ~15 cm backward — the human `foot` joint sits at the toe, ahead of the robot ankle. |
| 5 | **rot_offset [w,x,y,z]** | Fixed quaternion (**w-first / `scalar_first`**) **post-multiplied** onto the human orientation: `target_rot = human_rot × rot_offset`. A **frame-convention adapter** between human-joint axes and robot-link axes — not a routine tuning knob; change only to fix a systematic orientation mismatch. |

## Reading the asimov (hybrid) config

- **Legs** (`hip 10/20`, `knee 10/10`): light position + moderate orientation, small `±0.03` lateral hip nudge — shapes the leg without over-constraining.
- **Feet** (`100/10` + `-0.15` offset): strong position lock, light orientation; the offset handles placement (devs-style).
- **Waist** (`0/10`): orientation-only — track torso twist, don't pin its position.
- **Shoulders** (`0/1`): essentially free; the arm is driven from the ends.
- **Elbows / wrists** (`30/1`, `100/1`): **position-tracked** — the fix that stopped G1-inherited elbow hyperextension (pin wrist/elbow *positions*, let orientation float).

## Tuning cheatsheet

| symptom | change |
|---|---|
| Foot floats / tilts | raise foot `pos_weight`, lower `rot_weight`, adjust `pos_offset`, or enable `--leveling 1` |
| Toe/hip too low | weight `hip`/`knee` position more |
| Arm hyperextends | prefer **position** weight on elbow/wrist over shoulder **orientation** weight |
| Limbs reach too far / short | adjust `human_scale_table` (down = pull in) — use `--overlay` |
| A link is systematically mis-rotated | that's the `rot_offset` quat, not the weights |

## Live tuning

```bash
# edit smplx_to_asimov_tune.json, then:
python scripts/retarget_and_view.py --clip LeftTurn03            # one clip, looped
python scripts/retarget_and_view.py --clip LeftTurn03 --overlay  # + SMPL-H skeleton overlay
python scripts/retarget_and_view.py --overlay --speed 0.3        # all KIT/12, slow-mo
```

The overlay draws the (scaled) SMPL-H joint skeleton on top of the robot so you can see
whether each human target lands on the corresponding robot link — the direct signal for
tuning `human_scale_table`.
