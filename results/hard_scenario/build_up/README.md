# Build-up: scene factors one at a time on L0_faint (plan, written before the runs)

**Goal.** A benchmark that shows what thermal adds on top of depth. Target success on test set (a): rgb about 0.3,
rgbd about 0.6, rgbdt 0.7 or more.

**Where we are** (500 demos, rgb, test set (a); `../easy_l1/README.md` step 1, `../level_sweep_500/README.md`,
`../thermal_value/README.md`):
- L0 0.96, and L0_faint (L0 + the faint, tinted glass of L1) 0.26.
- On the L1 scene, rgb is 0.04 even with the glass back at the XML alpha 0.2 (L1_alpha02). Easing the glass
  (L1e_alpha) or the scene a little (L1e_scene) leaves it at 0.02.
- So the scene factors of L1, not the faint glass, are what collapse RGB. In the L1 scene the gripper closes about
  10.5 cm from the glass (median), against 2.3 cm on L0_faint: the policy does not find the glass.
- On L2 + blob depth defects on the glass (L2g), rgbd falls to 0.30 and rgbdt to 0.40, from 0.84 each on L2.

**Plan.** Start from L0_faint and add the scene factors of `sim/configs/hard_base.yaml` one at a time, to find a
scene where rgb stays at about 0.3 (step 1). Then add blob depth defects on the glass to that scene (step 2).

| id | level | config | = L0_faint + |
|---|---|---|---|
| C1 | L0f_distract | `sim/configs/L0f_distract.yaml` (meta/ep_level 17) | 1-3 opaque distractors (train mug / box / can / bowl, held-out opaque_bottle / opaque_cup) |
| C2 | L0f_table | `sim/configs/L0f_table.yaml` (18) | tables solid / checker / wood / noise (held-out stripes / marble / white), colour jitter 0.1 |
| C3 | L0f_light | `sim/configs/L0f_light.yaml` (19) | light normal / dim / bright / backlight (held-out spot / dusk), intensity jitter 0.15, direction jitter 15° |
| C4 | L0f_scene | `sim/configs/L0f_scene.yaml` (20) | C1 + C2 with normal light: several objects on a non-uniform background |
| | `<B>_g` | `sim/configs/<B>_g.yaml` (21), made in step 2 | the chosen level B + blob depth defects on the glass (`depth_defects: blobs`) |

- **Thermal.** Every new level uses the thermal of the hard levels, so that later runs with thermal match them:
  `sim/configs/thermal/calibrated_v1_hard.yaml` (calibrated_v1 with crossover in 10% of the episodes), named as a
  string so that it replaces L0's v0 thermal as a whole. It has no effect on rgb policies. L0_faint itself keeps v0.
- **Not in any build-up level:** the liquid and the sky-blue table of `hard_base.yaml`.

## Checks before the runs

- **Config loading** (`test_set(level, "a_in_distribution")` and `"g_heldout_all"` for all four levels):
  - Against L0_faint, only the intended keys differ (besides `name`, `level` and thermal): C1 `n_distractors` (0, 0)
    → (1, 3); C2 `background` xml → the pools above and `background_jitter` 0 → 0.1; C3 `lighting` normal → the pools
    above, `light_jitter` 0 → 0.15, `light_dir_jitter` 0 → 15; C4 the keys of C1 and C2.
  - All four: alpha (0.05, 0.1), tint 0.05, no liquid, no look-alike, `depth_defects: dots`, no absorbers, table
    patches, edge dropout or RGB noise.
  - (a) draws from the training pools. (g) draws from the held-out pools of the factors each level varies; the
    others stay at L0's xml table and normal light.
  - Thermal in all four levels and both test sets: name calibrated_v1, pipeline raw, normalize centered,
    crossover_p 0.1, norm_range (-174, 404), equal to L1's thermal.
- **Teacher.** `python sim/teacher_check.py --n 50 --levels L0f_distract L0f_table L0f_light L0f_scene --workers 16`:
  50/50 on training seeds and 50/50 on test set (g) for all four levels, and no episode moved a distractor more
  than 2 cm (`teacher_check.txt`). Without `MUJOCO_GL=egl` the workers hang on GLFW (no display), so it ran with
  `MUJOCO_GL=egl` and the repository and `sim/` on `PYTHONPATH`.
- **First frames.** `levels_seed0.png` to `levels_seed3.png`, made with
  `python sim/figure_levels.py --seed <s> --levels L0_faint L0f_distract L0f_table L0f_light L0f_scene --out results/hard_scenario/build_up/levels_seed<s>.png`.
  Rows: the five levels with the same seed. Columns: RGB, depth and thermal at sensor resolution, then the 96×96 RGB
  the policy gets. The target glass is circled. The L0_faint row shows thermal with v0's per-frame min-max.

## Experiment plan

All runs: 500 demos (seeds 0-499), 101 epochs, `latest.ckpt` evaluated on test sets (a) and (g) with 50 episodes
each (the defaults of `scripts/run_experiment.sh`), one seed per run, `GROUP=build_up` (results in
`results/hard_scenario/build_up/<run>/`).

**Step 1: rgb on the four levels** (they can run at the same time).

First a smoke test: `EPOCHS=1 NTEST=2 NENVS=2 GROUP=smoke bash scripts/run_experiment.sh L0f_scene rgb 5` must run
to the end. Its results, its 5-demo zarr and its `data/runs` folder are then deleted.

| id | command |
|---|---|
| C1 | `GROUP=build_up bash scripts/run_experiment.sh L0f_distract rgb 500` |
| C2 | `GROUP=build_up bash scripts/run_experiment.sh L0f_table rgb 500` |
| C3 | `GROUP=build_up bash scripts/run_experiment.sh L0f_light rgb 500` |
| C4 | `GROUP=build_up bash scripts/run_experiment.sh L0f_scene rgb 500` |

**Step 2: glass depth defects on the chosen level B.**
- `sim/configs/<B>_g.yaml`: base `<B>.yaml`, level 21, `depth_defects: blobs`, registered in
  `sim/configs/manifest.json`. Blob defects on the glass only, with the defaults L2g uses (missing 0.1-0.9, wrong
  depth on 0-30% of the rest). No black / metal objects, table hole patches or edge dropout.
- Checks: the teacher succeeds in 100% of the episodes, and the depth renders show defects on the glass only
  (`sim/preview_glass_depth.py`).
- Four runs first: `<B>_g` × rgb, rgbd, rgbdt and rgbt.
- If there is room, two more: `<B>` × rgbd and rgbdt (the baseline without defects).
- If there is room after that, the probe:
  ```
  python sim/probe.py collect --levels <B> <B>_g --workers <cores>
  python sim/probe.py train --levels <B> <B>_g --epochs 25 --seeds 2 \
      --inputs rgb rgbd rgbdt rgbt depth thermal --out results/hard_scenario/build_up/probe.md
  ```

## Selection rule (fixed before the runs, not changed after seeing the results)

- **Range:** rgb success on test set (a) of 0.15-0.50.
- **If C4 is in the range, B = L0f_scene.**
- **If C4 is outside the range, B is the level of C1-C3 in the range whose rgb is closest to 0.30.** On a tie, C1
  comes before C2, and C2 before C3. B then lacks part of "several objects on a non-uniform background", and the
  report says so.
- **If all four are outside the range,** step 2 does not start. The results are reported with a proposal for what
  to try next.
- **When step 1 ends, a short report comes before step 2:** (a) / (g) of the four runs, the choice and the reason.

## How to read the results (`<B>_g`, test set (a))

| result | meaning |
|---|---|
| rgb 0.15-0.45, rgbd 0.45-0.70, rgbdt ≥ 0.70 | goal reached → propose a second seed and the L0 control |
| rgb and rgbd in their ranges, rgbdt < 0.70 | usable as the benchmark; the limit of the simple fusion → the next task is to raise rgbdt with a fusion method (`policy/cue-gating` or similar) |
| rgbd < 0.45 | the glass depth defects break learning even in a simpler scene → compare with rgbt and with `<B>` × rgbd to find the cause |
| rgbt − rgbdt ≥ 0.15 | direct evidence that broken depth drags the simple fusion down |

With 50 episodes, differences under about 0.15 are within noise; use
`python policy/summarize_runs.py compare <run A> <run B>` (Fisher exact test). Whatever the results, all runs are
reported.

## Step 1 results (2026-11-13)

The four runs ran from 02:18 to 07:30 KST, one per GPU, 2.9-4.1 h each including evaluation; none failed. The smoke
test ran to the end first and its outputs were deleted.

| id | level | (a) | (g) | (a) against L0_faint rgb 0.26 (13/50) |
|---|---|---|---|---|
| C1 | L0f_distract | 0.04 (2/50) | 0.02 (1/50) | p = 0.004 |
| C2 | L0f_table | 0.02 (1/50) | 0.00 (0/50) | p = 0.0008 |
| C3 | L0f_light | 0.00 (0/50) | 0.04 (2/50) | p = 1e-4 |
| C4 | L0f_scene | 0.00 (0/50) | 0.00 (0/50) | p = 1e-4 |

p: Fisher exact, two-sided. L0_faint rgb is `../easy_l1/L0_faint_rgb_500demos` (same plan).

**Selection.** All four are below 0.15, so by the rule above step 2 does not start (no `<B>_g`).

**What the runs show.**
- **Each factor alone takes rgb from 0.26 to the floor.** Distractors, tables and varied light each do it, and so
  do distractors and tables together. With the faint glass, rgb on these levels is where it is on L1 (0.02).
- **Where the gripper closes for the last time** (median distance to the target; within 8 cm):

  | run | (a) | (g) |
  |---|---|---|
  | L0_faint | 2.3 cm; 49/50 | 2.5 cm; 48/49 |
  | C1 L0f_distract | 5.9 cm; 32/50 | 7.9 cm; 25/50 |
  | C2 L0f_table | 10.4 cm; 17/50 | 10.4 cm; 13/50 |
  | C3 L0f_light | 11.6 cm; 15/49 | 9.2 cm; 21/50 |
  | C4 L0f_scene | 9.3 cm; 18/50 | 10.3 cm; 15/50 |

  - With varied tables or light, the policy does not find the glass, as on L1 (10.4-10.9 cm in
    `../easy_l1/README.md`): it closes away from every object in 0.58-0.74 of the episodes.
  - With distractors only, it gets closer (missed_grasp 0.46 / 0.40, other 0.32 / 0.48) but rarely succeeds, and it
    never lifts a distractor.
- **No condition stands out.** In C3, the normal-light episodes fail too (0/14); there, normal light still has the
  intensity and direction jitter. Every table and distractor count has 0-1 successes.
- **The rollouts during training stay at 0.00-0.08** in all four runs; L0_faint's rose to 0.28 by epoch 100.
- **The thermal change is not a confound.** The new levels also differ from L0_faint in thermal (v0 →
  calibrated_v1). L0_faint with `thermal/calibrated_v1_hard.yaml` gives the same first-frame RGB (all pixels equal),
  target and position as L0_faint on 30 seeds (training seeds 0-19, test seeds 100000-100009).

**Proposed next candidates (not run; waiting for approval).** The same build-up with the glass at the XML alpha 0.2
and no tint, i.e. on L0 instead of L0_faint. Only the two ends are known: L0 alone gives 0.96, and all the factors of
`hard_base.yaml` together give 0.04 (L1_alpha02). Each run pairs with the C run of the same factor, which shows how
much of each collapse comes from the faint glass. RGB only, same protocol and the same selection rule (D4, then
D1-D3).

| id | level | = L0 (glass alpha 0.2) + |
|---|---|---|
| D1 | L0_distract | 1-3 opaque distractors (as C1) |
| D2 | L0_table | the tables of C2 |
| D3 | L0_light | the light of C3 |
| D4 | L0_scene | D1 + D2 with normal light (as C4) |
