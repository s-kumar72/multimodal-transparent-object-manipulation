# Easy L1: a level where RGB reaches about 0.3 (plan, written before the runs)

**Goal.** A benchmark with success on test set (a) of about 0.3 for rgb, about 0.6 for rgbd and 0.8-0.9 for rgbdt.
If rgbdt stays below 0.8, the next step fills the gap with a fusion method that weighs thermal by its reliability.

**Where we are** (500 demos, test set (a); `../level_sweep_500/README.md`, `../thermal_value/README.md`):
- L0 rgb 0.96; L1 rgb 0.02, rgbd 0.82, rgbdt 0.84. RGB alone collapses at L1.
- L2g (L2 + blob depth defects on the glass): rgb 0.02, rgbd 0.30, rgbdt 0.40.

**Plan.** Ease L1 until RGB reaches about 0.3 (the "easy L1"), then rebuild L2g on top of it. What L1 adds to L0:
1-3 opaque distractors, varied tables (sky-blue included) and light, liquid, calibrated thermal, and a fainter glass
(alpha 0.05-0.1, tint 0.05).

| level | config | change |
|---|---|---|
| L1e_alpha | `sim/configs/L1e_alpha.yaml` (meta/ep_level 14) | L1 with the glass alpha 0.05-0.1 → 0.1-0.15 (the XML alpha is 0.2); nothing else |
| L1e_scene | `sim/configs/L1e_scene.yaml` (15) | L1 with 0-2 distractors instead of 1-3 and no sky-blue table among the training backgrounds (the held-out pool is unchanged); the faint glass stays |
| L2g_easy | `sim/configs/L2g_easy.yaml` (16), made in step 2 | the chosen L1e_* + a look-alike in every episode (`lookalike_p: 1.0`) + blob depth defects (`depth_defects: blobs`). No black / metal objects, table holes or edge dropout, so the defects are on the glass only: the same construction as L2g on L2 |

## Checks before the runs

- **Config loading.** Both levels load for the training split and for test sets (a) and (g). Against L1, only the
  intended keys differ (besides `name` and `level`):
  - L1e_alpha: `alpha` (0.05, 0.1) → (0.1, 0.15);
  - L1e_scene: `n_distractors` (1, 3) → (0, 2), and the training background pool loses `skyblue`.
  - (a) draws from the training pools. (g) draws from the held-out pools, the same for both levels: tables stripes /
    marble / white, light spot / dusk, distractors opaque_bottle / opaque_cup.
- **Teacher.** `python sim/teacher_check.py --n 50 --levels L1e_alpha L1e_scene --workers 16`: 50/50 on training
  seeds and 50/50 on test set (g) for both levels (`teacher_check.txt`).
- **First frames.** `levels_seed0.png` to `levels_seed3.png`, made with
  `python sim/figure_levels.py --seed <s> --levels L1 L1e_alpha L1e_scene --out results/hard_scenario/easy_l1/levels_seed<s>.png`.
  - Rows: L1, L1e_alpha and L1e_scene with the same seed. Columns: RGB, depth and thermal at sensor resolution, then
    the 96×96 RGB the policy gets. The target glass is circled.
  - L1e_scene draws its distractors and table from its own pools, so its scene differs from L1's for the same seed.

## Experiment plan

All runs: 500 demos (seeds 0-499), 101 epochs, `latest.ckpt` evaluated on test sets (a) and (g) with 50 episodes
each, one seed per run, `GROUP=easy_l1` (results in `results/hard_scenario/easy_l1/<run>/`).

**Step 1: RGB only, four runs** (they can run at the same time). Each is
`GROUP=easy_l1 GPU=<n> bash scripts/run_experiment.sh <level> rgb 500`.

First a smoke test: `EPOCHS=1 NTEST=2 NENVS=2 GROUP=smoke bash scripts/run_experiment.sh L1e_scene rgb 5` must run
to the end. Its results, its 5-demo zarr and its `data/runs` folder are then deleted.

| id | level | purpose |
|---|---|---|
| R1 | L0_faint | effect of the faint glass alone on RGB (compare L0 rgb 0.96) |
| R2 | L1_alpha02 | RGB on L1 without the faint glass (compare L1 rgb 0.02) |
| E1 | L1e_alpha | candidate 1: medium transparency |
| E2 | L1e_scene | candidate 2: simpler scene |

**Step 2: L2g_easy on the chosen easy L1.**
- Make `sim/configs/L2g_easy.yaml` (base: the chosen `L1e_*.yaml`, name L2g_easy, level 16, `lookalike_p: 1.0`,
  `depth_defects: blobs`) and register it in `sim/configs/manifest.json`.
- Check that the teacher succeeds in 100% of the episodes, and that the depth renders show defects on the glass
  only (`sim/preview_glass_depth.py`).
- Four runs: L2g_easy × rgb, rgbd, rgbdt and rgbt.
  - rgbt is RGB + thermal without depth.
  - It is the baseline for whether broken depth drags the policy down, and the reference for the fusion method of
    the next step.
- If there is time (the policy runs come first), the probe:
  ```
  python sim/probe.py collect --levels L1e_alpha L1e_scene L2g_easy --workers <cores>
  python sim/probe.py train --levels L1e_alpha L1e_scene L2g_easy --epochs 25 --seeds 2 \
      --inputs rgb rgbd rgbdt rgbt depth thermal --out results/hard_scenario/easy_l1/probe.md
  ```

## Selection rule (fixed before the runs, not changed after seeing the results)

- **Only E1 (L1e_alpha) and E2 (L1e_scene) are candidates.** R1 and R2 explain causes and are not candidates.
- **The easy L1 is the candidate whose RGB success on test set (a) is closer to 0.30.** On a tie, E1 is chosen,
  since it changes one factor only.
- **If neither candidate is within 0.15-0.50** (each is below 0.15 or above 0.50), step 2 does not start. The
  results are reported with a proposal for the next candidates.
- **When step 1 ends, a short report comes before step 2:** (a) / (g) of the four runs, the choice and the reason.

## How to read the results (L2g_easy, test set (a))

| result | meaning |
|---|---|
| rgb 0.2-0.4, rgbd 0.5-0.7, rgbdt ≥ 0.8 | goal reached → confirm with a second seed |
| rgb 0.2-0.4, rgbd 0.5-0.7, rgbdt < 0.8 | usable as the benchmark → the next step's task is to raise rgbdt with the fusion method |
| rgbd < 0.4 | depth defects on the glass break learning even on the easier scene → compare with rgbt to find the cause |
| rgbt > rgbdt by 0.15 or more | direct evidence that broken depth drags the simple fusion down |
| rgbd ≈ rgbdt ≈ rgb | the policy uses neither depth nor thermal → report and discuss |

With 50 episodes, differences under about 0.15 are within noise; use
`python policy/summarize_runs.py compare <run A> <run B>` (Fisher exact test). Whatever the results, all runs are
reported.

## Step 1 results (2026-11-12)

| id | level | (a) | (g) | (a) compared with |
|---|---|---|---|---|
| R1 | L0_faint | 0.26 (13/50, CI 0.16-0.40) | 0.32 (16/50) | L0 rgb 0.96: p = 1e-13 |
| R2 | L1_alpha02 | 0.04 (2/50) | 0.00 (0/50) | L1 rgb 0.02: p = 1 |
| E1 | L1e_alpha | 0.02 (1/50) | 0.04 (2/50) | L1 rgb: p = 1 |
| E2 | L1e_scene | 0.02 (1/50) | 0.08 (4/50) | L1 rgb: p = 1 |

p: Fisher exact, two-sided. Also R1 vs L1 rgb p = 0.0008, R1 vs R2 p = 0.004.

**Selection.** Both candidates are below 0.15, so by the rule above step 2 does not start (no `L2g_easy`).

**What the runs show.**
- The faint glass alone takes RGB from 0.96 to 0.26 (R1). The shared scene of L1-L4 alone (`hard_base.yaml`, with
  the glass at the XML alpha 0.2 and no tint) takes it to 0.04 (R2). The scene is the bigger factor, and with it
  neither a less faint glass (E1) nor a slightly simpler scene (E2) moves RGB off the L1 floor.
- Where the gripper closes for the last time (distance to the target, test set (a), all 50 episodes closed):

  | run | median distance | within 8 cm |
  |---|---|---|
  | L0 rgb | 0.3 cm | 50/50 |
  | R1 L0_faint | 2.3 cm | 49/50 |
  | R2, E1, E2, L1 rgb | 10.4-10.9 cm | 15-17/50 |

  On L0_faint the policy finds the glass and misses it by a little (missed_grasp 0.42, wrong_height 0.20). In the
  L1 scene it closes away from every object (`other` 0.62-0.68): it does not locate the glass.
- E2's episodes with no distractor fail too (1/16), so the distractors alone are not the cause. No table, light or
  liquid condition has more than 2 successes in R2, E1 or E2, which does not single out one factor.
- R1's rollouts on (a) were still rising at epoch 100 (0.14, 0.22, 0.28 at epochs 50, 75, 100); the other three
  stayed at 0.00-0.04.

**Proposed next candidates (not run; waiting for approval).** Build up from L0_faint (0.26) instead of easing L1
down: one factor of `hard_base.yaml` per level, RGB only, same protocol as step 1.

| id | level | = L0_faint + |
|---|---|---|
| C1 | L0f_distract | 1-3 opaque distractors |
| C2 | L0f_table | the table pool solid / checker / wood / noise with jitter 0.1 (no sky-blue) |
| C3 | L0f_light | the light pool normal / dim / bright / backlight with its jitters |
| C4 | L0f_scene | C1 + C2 with normal light: the two scene properties the main scenario needs |
