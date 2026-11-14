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

## Round 2 plan (alpha 0.2)

Written before the runs (2026-11-14), after the user approved the proposal above.

**Why.** On L0_faint (rgb 0.26), adding any one scene factor took rgb to 0.00-0.04 (C1-C4). Round 2 builds the same
levels on L0 instead, with the glass as in L0 (the XML rgba 0.8 / 0.9 / 1.0, alpha 0.2, no tint). The known ends: L0
rgb 0.96 (`../level_sweep_500/L0_rgb_500demos`), and all the factors of `hard_base.yaml` together 0.04
(`../easy_l1/L1_alpha02_rgb_500demos`). Each Dk pairs with the Ck of the same factor, so Dk vs Ck is the share of the
faint glass in that collapse.

| id | level | config | = L0 + | pair |
|---|---|---|---|---|
| D1 | L0_distract | `sim/configs/L0_distract.yaml` (meta/ep_level 22) | 1-3 opaque distractors, as C1 | C1 0.04 |
| D2 | L0_table | `sim/configs/L0_table.yaml` (23) | the tables and jitter of C2 | C2 0.02 |
| D3 | L0_light | `sim/configs/L0_light.yaml` (24) | the light pools and jitters of C3 | C3 0.00 |
| D4 | L0_scene | `sim/configs/L0_scene.yaml` (25) | D1 + D2 with normal light, as C4 | C4 0.00 |
| | `<B>_g` | `sim/configs/<B>_g.yaml` (26), made in step 2 | the chosen level B + blob depth defects on the glass | |

All four name `thermal/calibrated_v1_hard.yaml` as a string, like C1-C4, and set neither `alpha` nor `tint`.

### Checks before the runs

- **Config loading** (`test_set(level, "a_in_distribution")` and `"g_heldout_all"`):
  - All four: `alpha` None (the XML rgba, alpha 0.2), `tint` 0; thermal name calibrated_v1, pipeline raw, normalize
    centered, crossover_p 0.1, norm_range (-174, 404), equal to L1's thermal. No liquid, look-alike, absorbers, table
    patches, edge dropout or RGB noise; `depth_defects: dots`.
  - Against L0, only the intended keys differ (besides `name`, `level` and thermal): D1 `n_distractors` (0, 0) →
    (1, 3); D2 the background pools and `background_jitter` 0.1; D3 the lighting pools, `light_jitter` 0.15 and
    `light_dir_jitter` 15; D4 the keys of D1 and D2.
  - Against its Ck, each Dk differs only in `alpha` ((0.05, 0.1) → None) and `tint` (0.05 → 0), on both test sets.
  - Pools: (a) distractors mug / box / can / bowl, tables solid / checker / wood / noise, light normal / dim / bright /
    backlight; (g) distractors opaque_bottle / opaque_cup, tables stripes / marble / white, light spot / dusk (each
    level only for the factors it varies; the rest stay at L0's xml table and normal light).
- **Same scene as the paired Ck except the glass** (first frame, training seeds 0-19 and seeds 100000-100009 of test
  sets (a) and (g), 40 seeds per pair, 160 in all):
  - Equal in every pair and seed: target object, the position of every body, episode info (background, light,
    distractors, temperatures, crossover; all keys but `level` and `alpha`), segmentation, depth and thermal (raw and
    the 96×96 policy inputs) and joints.
  - RGB differs only on the glass: every differing pixel of the 320×240 render is on the glass or within 1 pixel of its
    segmentation outline (the anti-aliased edge; at most 22-28 levels off the glass, checked on the training seeds of
    all four pairs).
  - The faint glass of Ck draws alpha 0.051-0.099 on these seeds.
- **Teacher.** `python sim/teacher_check.py --n 50 --levels L0_distract L0_table L0_light L0_scene --workers 16`:
  50/50 on training seeds and 50/50 on test set (g) for all four levels, and no episode moved a distractor more than
  2 cm (`teacher_check_round2.txt`; run with `MUJOCO_GL=egl` as in round 1).
- **First frames.** `levels_round2_seed0.png` to `levels_round2_seed3.png`, made with
  `python sim/figure_levels.py --seed <s> --levels L0 L0_distract L0_table L0_light L0_scene --out results/hard_scenario/build_up/levels_round2_seed<s>.png`.
  Same layout as round 1. The L0 row shows v0 thermal (per-frame min-max); the other rows the calibrated thermal.

### Experiment plan

All runs as in round 1: 500 demos (seeds 0-499), 101 epochs, `latest.ckpt` evaluated on test sets (a) and (g) with 50
episodes each (the defaults of `scripts/run_experiment.sh`), one seed per run, `GROUP=build_up`.

**Step 1: rgb on the four levels** (they can run at the same time, one per GPU, `GPU=<n>` in front).

First a smoke test: `EPOCHS=1 NTEST=2 NENVS=2 GROUP=smoke bash scripts/run_experiment.sh L0_scene rgb 5` must run to
the end. Its results, its 5-demo zarr and its `data/runs` folder are then deleted.

| id | command | pair |
|---|---|---|
| D1 | `GROUP=build_up bash scripts/run_experiment.sh L0_distract rgb 500` | C1 0.04 |
| D2 | `GROUP=build_up bash scripts/run_experiment.sh L0_table rgb 500` | C2 0.02 |
| D3 | `GROUP=build_up bash scripts/run_experiment.sh L0_light rgb 500` | C3 0.00 |
| D4 | `GROUP=build_up bash scripts/run_experiment.sh L0_scene rgb 500` | C4 0.00 |

Each finished run's results folder is committed and pushed to `exp/demo-scale`.

**Step 2: glass depth defects on the chosen level B** (only if step 1 chooses a B).
- `sim/configs/<B>_g.yaml`: base `<B>.yaml`, level 26, `depth_defects: blobs`, registered in
  `sim/configs/manifest.json`. Blob defects on the glass only, with the defaults (missing 0.1-0.9, wrong depth on 0-30%
  of the rest). No black / metal objects, table hole patches or edge dropout.
- Checks: the teacher succeeds in 100% of the episodes, and the depth renders show defects on the glass only
  (`sim/preview_glass_depth.py`).
- Four runs first: `<B>_g` × rgb, rgbd, rgbdt and rgbt.
- If there is room, two more: `<B>` × rgbd and rgbdt (the baseline without defects).

### Selection rule (fixed before the runs, not changed after seeing the results)

- **Range:** rgb success on test set (a) of 0.15-0.50.
- **If D4 is in the range, B = L0_scene.**
- **If D4 is outside the range, B is the level of D1-D3 in the range whose rgb is closest to 0.30.** On a tie, D1
  comes before D2, and D2 before D3. B then lacks part of "several objects on a non-uniform background", and the
  report says so.
- **If all four are outside the range,** step 2 does not start. The results are reported with a proposal for what to
  try next.
- **When step 1 ends, a short report comes before step 2:** (a) / (g) of the four runs, the comparison with the paired
  Ck, the choice and the reason.

### How to read the results (`<B>_g`, test set (a))

| result | meaning |
|---|---|
| rgb 0.15-0.45, rgbd 0.45-0.70, rgbdt ≥ 0.70 | goal reached → propose a second seed and the L0 control |
| rgb and rgbd in their ranges, rgbdt < 0.70 | usable as the benchmark; the limit of the simple fusion → the next task is to raise rgbdt with a fusion method |
| rgbd < 0.45 | the glass depth defects break learning → compare with rgbt and with `<B>` × rgbd to find the cause |
| rgbt − rgbdt ≥ 0.15 | direct evidence that broken depth drags the simple fusion down |

With 50 episodes, differences under about 0.15 are within noise; use
`python policy/summarize_runs.py compare <run A> <run B>` (Fisher exact test). With many comparisons, a p near 0.05
can be chance. Whatever the results, all runs are reported.

## Round 2 results

### Step 1: rgb on D1-D4 (2026-11-14)

The four runs ran from 10:31 to 17:44 KST, one per GPU. Two attempts failed with CUDA out of memory when other users'
jobs filled the GPU (the first smoke test at 10:20, and D4's first attempt at 15:01, right after its demos were
collected); both were rerun. The smoke test then ran to the end and its outputs were deleted.

| id | level | (a) (95% CI) | (g) | paired Ck (a) | p (a) vs Ck | p (a) vs L0 rgb 0.96 |
|---|---|---|---|---|---|---|
| D1 | L0_distract | 0.20 (10/50, 0.11-0.33) | 0.24 (12/50) | C1 0.04 (2/50) | 0.028 | 9e-16 |
| D2 | L0_table | 0.00 (0/50, 0.00-0.07) | 0.04 (2/50) | C2 0.02 (1/50) | 1 | 3e-26 |
| D3 | L0_light | 0.04 (2/50, 0.01-0.13) | 0.04 (2/50) | C3 0.00 (0/50) | 0.495 | 3e-23 |
| D4 | L0_scene | 0.04 (2/50, 0.01-0.13) | 0.04 (2/50) | C4 0.00 (0/50) | 0.495 | 3e-23 |

p: Fisher exact, two-sided. L0 rgb is `../level_sweep_500/L0_rgb_500demos` (48/50 on (a), 47/50 on (g)).

**Selection.** D4 is below 0.15, so B is chosen among D1-D3. Only D1 is in 0.15-0.50, so **B = L0_distract**. B has
several objects but L0's plain table: it lacks the non-uniform background of "several objects on a non-uniform
background". The full analysis follows when step 2 ends.

### Step 2 checks

- **`sim/configs/L0_distract_g.yaml`** (level 26): base `L0_distract.yaml`, `depth_defects: blobs`, registered in the
  manifest. On test sets (a) and (g), it differs from L0_distract only in `name`, `level` and `depth_defects`
  (dots → blobs); its defect settings equal L2g's (missing 0.1-0.9, blob size 1.5-5 px, wrong depth on 0-30% of the
  rest with offsets 0.05-0.3 m, no see-through episodes). No absorbers, table patches or edge dropout.
- **Teacher.** `python sim/teacher_check.py --n 50 --levels L0_distract_g --workers 16`: 50/50 on training seeds and
  50/50 on test set (g), no distractor moved more than 2 cm (`teacher_check_round2_g.txt`).
- **Depth on the glass only.** `python sim/preview_glass_depth.py --levels L0_distract L0_distract_g --seeds 0 1 2 3 4 5 6 7 --out results/hard_scenario/build_up/glass_depth_round2.png`:
  - On L0_distract_g, 0.13-0.88 of the glass pixels are missing and 0.03-0.24 have wrong depth, in connected blobs.
  - On L0_distract (dots), 0.37-0.50 of the glass pixels are missing, scattered, and none are wrong.
  - Off the glass, no pixel is missing on any seed of either level.

**Step 2 runs** (as planned): `L0_distract_g` × rgb, rgbd, rgbdt and rgbt first, then `L0_distract` × rgbd and
rgbdt when a GPU is free.
