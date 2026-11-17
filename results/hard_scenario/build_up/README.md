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

All ten runs ran on 2026-11-14, from 10:31 to 23:38 KST (2.0-3.2 h each including demo collection and evaluation, up to four at once).
p is Fisher exact, two-sided, from `python policy/summarize_runs.py compare`; 95% CIs are Wilson. With 50 episodes,
differences under about 0.15 are within noise. This section makes more than 40 comparisons, so a p near 0.05 can be
chance.

### Failed attempts

Three attempts failed, all with `torch.OutOfMemoryError: CUDA out of memory` when other users' jobs (9.5 GiB
processes) filled the GPU. Each was rerun and finished; the 500-demo sets were reused. The logs are in the git-ignored
`data/runs/`.

| when (KST) | attempt | log | rerun |
|---|---|---|---|
| 10:20 | smoke test (`L0_scene rgb 5`) on GPU 3, in training; the GPU had about 8 GB free | runner line in `build_up2.log`, start of the run in `smoke_build2_oom1.log` (its training log was deleted with the smoke outputs, as planned) | started again requiring 12 GB free; ran to the end at 10:30 |
| 15:01 | D4 `L0_scene rgb 500`, attempt 1, GPU 3, right after its demos were collected | `L0_scene_rgb_500demos.log`: "Tried to allocate 80.00 MiB. GPU 0 has a total capacity of 47.40 GiB of which 96.00 MiB is free" | 15:17 → done 17:44 |
| 17:53 | `L0_distract_g rgbdt 500`, attempt 1, GPU 2 | `L0_distract_g_rgbdt_500demos.log`: "Tried to allocate 20.00 MiB ... of which 49.31 MiB is free" | 17:56 → done 21:09 |

### All runs

From `python policy/summarize_runs.py table` on the ten folders (full output with the training rollouts and success
by condition: `round2_table.md`). Outcomes are shares of the 50 episodes; wrong_object and timeout are at most 0.02
in every run.

| run | (a) (95% CI) | (g) (95% CI) | (a) missed_grasp / other / dropped / wrong_height | (g) missed_grasp / other / dropped / wrong_height |
|---|---|---|---|---|
| D1 L0_distract rgb | 0.20 (10/50, 0.11-0.33) | 0.24 (12/50, 0.14-0.37) | 0.56 / 0.02 / 0.02 / 0.20 | 0.46 / 0.06 / 0.06 / 0.16 |
| D2 L0_table rgb | 0.00 (0/50, 0.00-0.07) | 0.04 (2/50, 0.01-0.13) | 0.36 / 0.52 / 0.04 / 0.08 | 0.20 / 0.70 / 0.06 / 0.00 |
| D3 L0_light rgb | 0.04 (2/50, 0.01-0.13) | 0.04 (2/50, 0.01-0.13) | 0.18 / 0.62 / 0.14 / 0.02 | 0.26 / 0.66 / 0.02 / 0.02 |
| D4 L0_scene rgb | 0.04 (2/50, 0.01-0.13) | 0.04 (2/50, 0.01-0.13) | 0.22 / 0.68 / 0.04 / 0.02 | 0.20 / 0.74 / 0.02 / 0.00 |
| L0_distract_g rgb | 0.18 (9/50, 0.10-0.31) | 0.34 (17/50, 0.22-0.48) | 0.50 / 0.06 / 0.06 / 0.18 | 0.44 / 0.02 / 0.06 / 0.12 |
| L0_distract_g rgbd | 0.46 (23/50, 0.33-0.60) | 0.44 (22/50, 0.31-0.58) | 0.40 / 0.00 / 0.00 / 0.14 | 0.36 / 0.04 / 0.10 / 0.06 |
| L0_distract_g rgbdt | 0.54 (27/50, 0.40-0.67) | 0.38 (19/50, 0.26-0.52) | 0.24 / 0.00 / 0.08 / 0.14 | 0.38 / 0.02 / 0.08 / 0.14 |
| L0_distract_g rgbt | 0.40 (20/50, 0.28-0.54) | 0.50 (25/50, 0.37-0.63) | 0.34 / 0.02 / 0.22 / 0.02 | 0.26 / 0.08 / 0.14 / 0.00 |
| L0_distract rgbd | 0.84 (42/50, 0.71-0.92) | 0.80 (40/50, 0.67-0.89) | 0.10 / 0.00 / 0.06 / 0.00 | 0.08 / 0.00 / 0.06 / 0.06 |
| L0_distract rgbdt | 0.88 (44/50, 0.76-0.94) | 0.88 (44/50, 0.76-0.94) | 0.04 / 0.00 / 0.02 / 0.06 | 0.04 / 0.00 / 0.04 / 0.04 |

### Step 1: rgb on D1-D4

| id | level | (a) (95% CI) | (g) | paired Ck (a) / (g) | difference Dk − Ck (a) / (g) | p vs Ck (a) / (g) | last close: median distance to the target; within 8 cm (a) / (g) |
|---|---|---|---|---|---|---|---|
| D1 | L0_distract | 0.20 (10/50, 0.11-0.33) | 0.24 (12/50) | C1 0.04 / 0.02 | +0.16 / +0.22 | 0.028 / 0.0018 | 2.8 cm; 49/50 / 2.4 cm; 46/50 |
| D2 | L0_table | 0.00 (0/50, 0.00-0.07) | 0.04 (2/50) | C2 0.02 / 0.00 | −0.02 / +0.04 | 1 / 0.495 | 8.3 cm; 24/50 / 12.0 cm; 12/50 |
| D3 | L0_light | 0.04 (2/50, 0.01-0.13) | 0.04 (2/50) | C3 0.00 / 0.04 | +0.04 / 0.00 | 0.495 / 1 | 11.4 cm; 15/50 / 9.4 cm; 17/50 |
| D4 | L0_scene | 0.04 (2/50, 0.01-0.13) | 0.04 (2/50) | C4 0.00 / 0.00 | +0.04 / +0.04 | 0.495 / 0.495 | 10.5 cm; 15/50 / 10.7 cm; 12/50 |

Against L0 rgb (`../level_sweep_500/L0_rgb_500demos`, 48/50 on (a), 47/50 on (g); last close 0.3 cm, 50/50 within
8 cm): D1 p = 9e-16 / 2e-13, D2 3e-26 / 5e-22, D3 and D4 3e-23 / 5e-22. For the Ck rows: C1 5.9 cm, 32/50 within
8 cm on (a); C2 10.4 cm, 17/50; C3 11.6 cm, 15/49; C4 9.3 cm, 18/50. In every run of this table every episode
closes the gripper at least once, except one episode of C3.

**Selection.** D4 is below 0.15, so B is chosen among D1-D3. Only D1 is in 0.15-0.50, so **B = L0_distract**. B has
several objects but L0's plain table: it lacks the non-uniform background of "several objects on a non-uniform
background".

**Which scene factor lowers rgb the most.**
- **Varied tables and varied light each take rgb from 0.96 to the floor** (D2 0.00 / 0.04, D3 0.04 / 0.04), and so
  do distractors and tables together (D4). The two cannot be ranked against each other (0/50 vs 2/50).
  - In these runs the policy does not find the glass: `other` (closed away from every object) is 0.52-0.74, and the
    last close is a median 8.3-12.0 cm from the target.
  - D2 has no distractors and still fails on every training table on (a) (checker 0/7, noise 0/16, solid 0/14, wood
    0/13); on (g) it succeeds once on stripes (1/18) and once on white (1/23), never on marble (0/9).
  - D3 succeeds only under backlight on (a) (2/11; bright 0/11, dim 0/14, normal 0/14, where normal light still
    has the intensity and direction jitter); on (g) 2/28 under spot and 0/22 at dusk.
  - D4 has no table, light or distractor count with more than 1 success.
- **Distractors alone lower rgb the least:** 0.96 → 0.20 / 0.24 (D1).
  - Here the policy finds the glass and misses it by a little: missed_grasp 0.56 / 0.46, wrong_height 0.20 / 0.16,
    other 0.02 / 0.06, last close 2.8 / 2.4 cm.
  - The count does not matter: 4/16, 3/19 and 3/15 for 1, 2 and 3 distractors on (a); 5/16 and 7/34 for 1 and 2
    on (g) (its pool has two distractors).
  - It never lifts a distractor on (a) (wrong_object 0.00; 0.02 on (g)).
- **The share of the faint glass.**
  - With distractors, the faint glass costs a further 0.16 / 0.22 (D1 vs C1, p = 0.028 / 0.0018). Of the drop from L0
    rgb to C1 (0.92 on (a)), 0.76 comes from the distractors and 0.16 from the faint glass. It also moves the last
    close from 2.8 to 5.9 cm (32/50 within 8 cm instead of 49/50).
  - With tables or varied light, the faint glass adds nothing measurable (Dk vs Ck p ≥ 0.495, and the last-close
    distances of Dk and Ck are alike): the factor alone collapses rgb.
  - So round 1's collapse of C2-C4 does not need the faint glass. C1's does: C1 is the one Ck the faint glass explains.

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
- **The two demo sets differ only in depth** (checked after the runs, every element of the 500-demo zarrs): rgb,
  thermal, thermal_raw, joints and actions are equal, and so is every meta array but `ep_level`. The test episodes
  also have the same object, scene and temperatures in all seven step 2 runs. So:
  - rgb and rgbt, which do not see depth, get the same training data on both levels. `L0_distract_g` rgbt is
    therefore also the `L0_distract` rgbt run.
  - The two rgb runs (D1 and `L0_distract_g` rgb) repeat one setup with the same data and config (training seed
    42). They give 10 vs 9 successes on (a) and 12 vs 17 on (g): the run-to-run spread of one setup is already
    0.10 on (g).

### Step 2 results: `L0_distract_g` (B with glass depth defects) and `L0_distract` (B without)

| input | (a) (95% CI) | (g) | target on (a) | met |
|---|---|---|---|---|
| rgb | 0.18 (9/50, 0.10-0.31) | 0.34 (17/50) | about 0.3 (0.15-0.45) | yes: in the range, below 0.3 |
| rgbd | 0.46 (23/50, 0.33-0.60) | 0.44 (22/50) | about 0.6 (0.45-0.70) | barely: one success above 0.45, below 0.6; (g) is below 0.45 |
| rgbdt | 0.54 (27/50, 0.40-0.67) | 0.38 (19/50) | ≥ 0.70 | no |
| rgbt | 0.40 (20/50, 0.28-0.54) | 0.50 (25/50) | (reference) | |
| without defects (`L0_distract`): rgbd | 0.84 (42/50, 0.71-0.92) | 0.80 (40/50) | | |
| without defects: rgbdt | 0.88 (44/50, 0.76-0.94) | 0.88 (44/50) | | |

**Comparisons** (successes out of 50; difference = second − first).

| comparison | (a) | p (a) | (g) | p (g) |
|---|---|---|---|---|
| rgb vs rgbd (`_g`) | 9 vs 23 (+0.28) | 0.005 | 17 vs 22 (+0.10) | 0.41 |
| rgbd vs rgbdt (`_g`; the value of thermal) | 23 vs 27 (+0.08) | 0.55 | 22 vs 19 (−0.06) | 0.69 |
| rgbt vs rgbdt (`_g`) | 20 vs 27 (+0.14) | 0.23 | 25 vs 19 (−0.12) | 0.31 |
| rgb vs rgbt (`_g`) | 9 vs 20 (+0.22) | 0.027 | 17 vs 25 (+0.16) | 0.16 |
| rgb vs rgbdt (`_g`) | 9 vs 27 (+0.36) | 3e-4 | 17 vs 19 (+0.04) | 0.84 |
| rgb: `L0_distract` vs `_g` (same training data) | 10 vs 9 (−0.02) | 1 | 12 vs 17 (+0.10) | 0.38 |
| rgbd: `L0_distract` vs `_g` | 42 vs 23 (−0.38) | 1e-4 | 40 vs 22 (−0.36) | 4e-4 |
| rgbdt: `L0_distract` vs `_g` | 44 vs 27 (−0.34) | 3e-4 | 44 vs 19 (−0.50) | 3e-7 |
| rgbd vs rgbdt (`L0_distract`, no defects) | 42 vs 44 (+0.04) | 0.77 | 40 vs 44 (+0.08) | 0.41 |
| rgb vs rgbd (`L0_distract`) | 10 vs 42 (+0.64) | 1e-10 | 12 vs 40 (+0.56) | 3e-8 |
| rgb: L2g vs `_g` | 1 vs 9 (+0.16) | 0.016 | 2 vs 17 (+0.30) | 2e-4 |
| rgbd: L2g vs `_g` | 15 vs 23 (+0.16) | 0.15 | 18 vs 22 (+0.08) | 0.54 |
| rgbdt: L2g vs `_g` | 20 vs 27 (+0.14) | 0.23 | 18 vs 19 (+0.02) | 1 |

L2g is `../thermal_value/L2g_*_500demos` (the L2 scene: the faint glass, 1-3 distractors, varied tables and
light, liquid and a look-alike of the target in every episode, plus the same blob defects). It has no rgbt run.

**What the comparisons show.**
- **The glass depth defects are the main effect.** They cost rgbd 0.38 / 0.36 and rgbdt 0.34 / 0.50 (p ≤ 4e-4).
  rgb does not change (same data; 0.20 vs 0.18, 0.24 vs 0.34).
- **Thermal adds nothing measurable on top of depth**, with defects (+0.08 / −0.06) or without (+0.04 / +0.08).
- **Thermal without depth (rgbt) gets as far as depth with defects:** rgbt 0.40 / 0.50 against rgbd 0.46 / 0.44.
  Against rgb, rgbt is +0.22 on (a) (p = 0.027, a p that this many comparisons can produce by chance) and +0.16 on
  (g) (p = 0.16).
- **rgbdt does not combine the two:** 0.54 / 0.38, against the better single input 0.46 / 0.50. rgbt − rgbdt is
  −0.14 on (a) and +0.12 on (g), within noise either way.
- **The ordering rgb < rgbd < rgbdt holds on (a) only.** On (g), rgb 0.34, rgbd 0.44, rgbdt 0.38 and rgbt 0.50 do
  not separate (p ≥ 0.16). Part of this is rgb: 0.34 on (g) against 0.18 on (a) (p = 0.11), while the repeat rgb
  run gave 0.24 on (g).
- **Against L2g**, the simpler scene raises rgb (0.02 → 0.18, 0.04 → 0.34), as intended. rgbd and rgbdt rise by
  0.02-0.16, which does not separate from L2g (p ≥ 0.15). With the same glass defects, depth policies land at
  0.30-0.54 in both scenes.

**Learning curves.** Rollouts on (a) during training (50 episodes at each evaluation) and validation loss:

| run | ep0 | ep25 | ep50 | ep75 | ep100 | lowest val loss (epoch) | val loss at epoch 100 |
|---|---|---|---|---|---|---|---|
| `_g` rgb | 0.00 | 0.04 | 0.06 | 0.24 | 0.32 | 0.0025 (81) | 0.0031 |
| `_g` rgbd | 0.00 | 0.34 | 0.30 | 0.42 | 0.36 | 0.0025 (93) | 0.0026 |
| `_g` rgbdt | 0.00 | 0.36 | 0.36 | 0.46 | 0.48 | 0.0028 (87) | 0.0029 |
| `_g` rgbt | 0.00 | 0.16 | 0.42 | 0.34 | 0.46 | 0.0041 (66) | 0.0055 |
| `L0_distract` rgb (D1) | 0.00 | 0.04 | 0.12 | 0.24 | 0.30 | 0.0026 (81) | 0.0035 |
| `L0_distract` rgbd | 0.00 | 0.72 | 0.86 | 0.92 | 0.90 | 0.0017 (100) | 0.0017 |
| `L0_distract` rgbdt | 0.00 | 0.64 | 0.80 | 0.88 | 0.88 | 0.0024 (96) | 0.0027 |

- With defects, rgbd and rgbdt reach 0.34-0.36 by epoch 25 and stay at 0.30-0.48. Without defects they reach
  0.64-0.72 by epoch 25 and 0.88-0.90 by epoch 100.
- Both rgb runs rise slowly and were still rising at epoch 100 (0.24 → 0.30-0.32 from epoch 75).
- rgbt's validation loss is lowest at epoch 66 and 34% higher at epoch 100, while its rollouts do not fall (0.42 at
  epoch 50, 0.46 at 100). The other runs' lowest validation loss is at epochs 81-100.

**Failure types.**
- **Defects turn successes into near misses, not into search failures.** `other` is at most 0.08 in every step 2 run.
  With depth or thermal, the last close is a median 1.3-1.6 cm from the target (45-50/50 within 8 cm; rgb 2.5 /
  2.3 cm). Without defects it is 0.6 / 0.6 cm (rgbd) and 0.4 / 0.5 cm (rgbdt).
- **missed_grasp is the main failure** with defects: rgb 0.50 / 0.44, rgbd 0.40 / 0.36, rgbdt 0.24 / 0.38, rgbt
  0.34 / 0.26. Without defects: rgbd 0.10 / 0.08, rgbdt 0.04 / 0.04 (rgbd with vs without defects on (a): 20 vs 5,
  p = 0.001).
- rgbt drops the glass more often (dropped 0.22 / 0.14, against rgbdt 0.08 / 0.08; p = 0.09 on (a)) and is rarely
  at the wrong height (0.02 / 0.00, against rgbdt 0.14 / 0.14; p = 0.06 on (a)). Neither difference is
  significant alone.

**Success by condition** (successes / episodes).
- **Thermal crossover** (6 of the 50 episodes in each test set):

  | run | crossover (a) | crossover (g) | no crossover (a) | no crossover (g) |
  |---|---|---|---|---|
  | `_g` rgb | 2/6 | 2/6 | 7/44 | 15/44 |
  | `_g` rgbd | 4/6 | 2/6 | 19/44 | 20/44 |
  | `_g` rgbdt | 3/6 | 2/6 | 24/44 | 17/44 |
  | `_g` rgbt | 0/6 | 1/6 | 20/44 | 24/44 |
  | `L0_distract` rgbd | 5/6 | 5/6 | 37/44 | 35/44 |
  | `L0_distract` rgbdt | 5/6 | 6/6 | 39/44 | 38/44 |

  - rgbt succeeds in 1 of the 12 crossover episodes, against 44 of the 88 others (p = 0.011). Without depth, the
    policy fails where the glass's thermal contrast flips, which fits rgbt finding the glass by thermal.
  - rgbdt succeeds in 5 of the 12 crossover episodes (vs rgbt 1/12, p = 0.15); rgb, which sees no thermal, 4/12.
- **Distractor count:** no trend that repeats across runs. For example, rgbd on (a) gives 6/16, 5/19 and 12/15 for
  1, 2 and 3 distractors, and rgbdt 9/16, 7/19 and 11/15.
- **Objects:** the wineglass fails in every rgb episode (0/10 on each test set in both rgb runs, 0/40). With depth or
  thermal it succeeds in 0.40-0.70 of its episodes (`_g`), and in 0.70-0.80 without defects.

### Conclusion

**Where `L0_distract_g` falls in the table "How to read the results" (test set (a)):** rgb 0.18 is in 0.15-0.45
and rgbd 0.46 in 0.45-0.70, and rgbdt 0.54 is below 0.70. That is the second row: **usable as the benchmark; the limit
of the simple fusion**. The goal (first row) is not reached. The other rows:
- "rgbd < 0.45": does not apply, but only by one success (23/50; 22/50 on (g)).
- "rgbt − rgbdt ≥ 0.15": does not apply (−0.14 on (a), +0.12 on (g)). There is no direct evidence that broken depth
  drags the simple fusion below rgbt. Rather, rgbdt fails to add thermal to depth.

**Caveats for using it as the benchmark.**
- The ordering rgb < rgbd < rgbdt holds on (a) only; on (g) none of the four inputs separates.
- rgbd sits on the edge of its range.
- One run per input, and the repeated rgb setup already moved by 0.10 on (g).
- B lacks the non-uniform background, because varied tables alone take rgb to the floor even with the glass at
  alpha 0.2 (D2).

**Proposed next steps (not run; waiting for approval).**
1. **A second training seed for the four `L0_distract_g` runs**: `EXTRA=training.seed=43 TAG=seed43`, no code
   change. One round on four GPUs, about 3 h (the demos exist). It shows whether the ordering on (a) and rgbd's place in its range
   hold before a fusion method is built on this benchmark.
2. **Then the fusion method that the second row of the table calls for**, judged on `L0_distract_g` rgbdt (now 0.54
   / 0.38; target ≥ 0.70) against rgbt (0.40 / 0.50) and the rgbdt without defects (0.88 / 0.88).
   - The aim: a policy that falls back on thermal where depth on the glass is broken and on depth where thermal flips
     (crossover).
   - Candidates: modality dropout in training (randomly removing depth or thermal), a missing-depth mask as an input,
     or separate encoders per modality with a learned gate.
   - These change `policy/`, so the design is reported for approval before any code.
