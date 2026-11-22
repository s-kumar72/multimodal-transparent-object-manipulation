# Benchmark: rgb about 0.4 and rgbd about 0.6, set by the environment (plan, written before the runs)

**Goal.** A benchmark level where the environment sets rgb at about 0.4 and rgbd at about 0.6. The simple fusion
rgbdt is expected at about 0.65 and is reported as the baseline. rgbdt 0.8-0.9 is the target for a later fusion
method, so this round does not try to reach it through the environment.

**Where we are** (`../build_up/README.md`, "Round 2 results"). L0_distract is L0 plus 1-3 opaque distractors, with the
glass at the XML alpha 0.2. 500 demos, success on test sets (a) / (g):
- no depth defects: rgb 0.20 / 0.24, rgbd 0.84 / 0.80, rgbdt 0.88 / 0.88;
- blob depth defects on the glass (L0_distract_g): rgb 0.18 / 0.34, rgbd 0.46 / 0.44, rgbdt 0.54 / 0.38, rgbt
  0.40 / 0.50.

The blob defects spoil grasp precision (missed_grasp 0.24-0.50, against 0.04-0.10 without them), and thermal does not
win it back. So this round uses two other levers:
- **rgb about 0.4:** a less transparent glass (alpha 0.3 or 0.4).
- **rgbd about 0.6:** in some episodes, depth sees through the glass: the glass pixels hold the depth of the table
  behind, with no holes, as a time-of-flight camera like the RealSense L515 reports. The other episodes keep
  L0_distract's normal depth (the existing dot holes on the glass). Expected: rgbd ≈ (1 − p) × 0.84 + p × rgb.

## Implementation (step 0)

**See-through in dot mode** (`sim/sensors.py`). Until now `glass_seethrough_p` worked only with `depth_defects: blobs`.
- `Sensors.episode()` draws the see-through flag whenever `glass_seethrough_p > 0`, in either mode, and stores it in
  the scene. With p = 0 nothing is drawn, so existing levels use the same random numbers as before. In blob mode the
  draw stays where it was.
- In the dot path of `render()`, a see-through episode keeps the depth of the surface behind on the glass pixels and
  makes no holes there. The dot draw (`rng.random`) is still made every frame, so the later draws keep their order.
- Blob mode (L2gs and the others) is unchanged.
- `episode_info["glass_seethrough"]` comes from the scene in both modes. The comments in `sim/config.py` and
  `sim/env.py` now say so; nothing else changed there.

**Regression check** (`sim/regression_test.py`; 190 arrays per config: 20 first frames with a random and with a fixed
object, rgb / depth / thermal / joints, and 5 whole teacher episodes with their actions and success):
- Baselines saved with the code before the change: `--save data/regress/pre_<level>.npz --config <level>.yaml` for
  L2gs, L3b and L4b, and also L0_distract and L0_distract_g (the base of the new levels and its blob version).
- Before the change, `--check results/data/regress_main.npz` (the default config): 190/190.
- After the change, all six give **190/190 arrays identical**: the default config, L2gs, L3b, L4b, L0_distract and
  L0_distract_g.

**New levels** (all with base `L0_distract.yaml`, registered in `sim/configs/manifest.json`):

| id | level | config | change from L0_distract | stands in for |
|---|---|---|---|---|
| A3 | L0_distract_a03 | `sim/configs/L0_distract_a03.yaml` (meta/ep_level 27) | `alpha: [0.3, 0.3]`, `tint: 0.0` | glass that is easier to see in RGB |
| A4 | L0_distract_a04 | `sim/configs/L0_distract_a04.yaml` (28) | `alpha: [0.4, 0.4]`, `tint: 0.0` | the same, more so |
| S5 | L0_distract_s50 | `sim/configs/L0_distract_s50.yaml` (29) | `glass_seethrough_p: 0.5` (glass at alpha 0.2) | a time-of-flight camera that reports the surface behind the glass in half of the episodes. The L515 of TRansPose has a reading on 86% of the glass pixels, and 95% of them are more than 2 cm off (median 132 mm; `real_data/README.md`) |

## Checks before the runs

- **Config loading** (`test_set(level, "a_in_distribution")` and `"g_heldout_all"`): against L0_distract, each level
  differs only in `name`, `level` and its intended key: A3 `alpha` None → (0.3, 0.3); A4 None → (0.4, 0.4); S5
  `glass_seethrough_p` 0 → 0.5.
  - All three keep `tint` 0, `depth_defects: dots`, `glass_holes` 0.4, 1-3 distractors, the xml table and normal light.
  - Thermal is calibrated_v1 with crossover_p 0.1. There is no look-alike, absorber, table patch, edge dropout or RGB
    noise.
- **Same scene as L0_distract** (first frame of training seeds 0-19 and of seeds 100000-100009 of test sets (a) and
  (g), 40 seeds per level):
  - Equal for every seed:
    - the target and the position of every body;
    - episode info (all keys but `level`, `alpha` and `glass_seethrough`);
    - segmentation, raw thermal, the 96×96 thermal and the joints.
  - A3, A4:
    - depth (320×240 and 96×96) is equal, and `glass_seethrough` is 0;
    - RGB differs only on the glass and its 1-pixel anti-aliased edge, with no differing pixel beyond it;
    - alpha is 0.300 / 0.400 on every seed.
  - S5: RGB (320×240 and 96×96) is equal everywhere.
    - **Normal episodes:** depth is equal to L0_distract's.
    - **See-through episodes:** depth differs only on the glass, and only where L0_distract has a dot hole. Every
      glass pixel holds the depth of the surface behind, and none is missing.
    - See-through in 12 of training seeds 0-19 and in 3 of test seeds 100000-100009.
  - Over all seeds: see-through in 244 of the 500 training seeds and 22 of the 50 test seeds. The test sets share
    their seeds, so (a) and (g) each have the same 22 see-through and 28 normal episodes.
- **Teacher.** `python sim/teacher_check.py --n 50 --levels L0_distract_a03 L0_distract_a04 L0_distract_s50 --workers 16`
  gives, for all three levels:
  - 50/50 on training seeds and 50/50 on test set (g);
  - no episode moved a distractor more than 2 cm (`teacher_check.txt`).
- **Depth of S5.** `python sim/preview_glass_depth.py --levels L0_distract L0_distract_s50 --seeds 0 1 2 3 4 5 6 7 --out results/hard_scenario/benchmark/glass_depth_s50.png`:
  - **See-through seeds (0, 1, 2, 6, 7):** no glass pixel is missing or wrong; the glass shows only as its cyan
    outline.
  - **Normal seeds (3, 4, 5):** 0.38 / 0.50 / 0.44 of the glass pixels are missing, the same as L0_distract on those
    seeds (dot holes).
  - No pixel off the glass is missing in either level.
- **First frames.** `alpha_seed0.png` to `alpha_seed3.png` compare the glass at alpha 0.2 / 0.3 / 0.4, made with
  `python sim/figure_levels.py --seed <s> --levels L0_distract L0_distract_a03 L0_distract_a04 --out results/hard_scenario/benchmark/alpha_seed<s>.png`.
  - Rows: L0_distract, A3 and A4 with the same seed.
  - Columns: RGB, depth and thermal at sensor resolution, then the 96×96 RGB the policy gets.

## Experiment plan

All runs: 500 demos (seeds 0-499), 101 epochs, `latest.ckpt` evaluated on test sets (a) and (g) with 50 episodes
each (the defaults of `scripts/run_experiment.sh`), `GROUP=benchmark` (results in `results/hard_scenario/benchmark/<run>/`).
Each finished run's results folder is committed to `exp/demo-scale`.

**Step 1: measure the levers (four runs at once).** First a smoke test:
`EPOCHS=1 NTEST=2 NENVS=2 GROUP=smoke bash scripts/run_experiment.sh L0_distract_s50 rgbd 5` must run to the end. Its
results, its 5-demo zarr and its `data/runs` folder are then deleted.

| id | command | purpose |
|---|---|---|
| A3 | `GROUP=benchmark bash scripts/run_experiment.sh L0_distract_a03 rgb 500` | rgb at alpha 0.3 |
| A4 | `GROUP=benchmark bash scripts/run_experiment.sh L0_distract_a04 rgb 500` | rgb at alpha 0.4 |
| S5d | `GROUP=benchmark bash scripts/run_experiment.sh L0_distract_s50 rgbd 500` | rgbd with see-through in half of the episodes |
| S5t | `GROUP=benchmark bash scripts/run_experiment.sh L0_distract_s50 rgbdt 500` | how much thermal fills in the see-through episodes |

**Step 2: the benchmark level** (after the selection below).
- `sim/configs/L0_bench.yaml`: base `L0_distract.yaml`, level 30, `alpha: [alpha*, alpha*]`, `tint: 0.0`,
  `glass_seethrough_p: p*`. If alpha* is 0.2, `alpha` and `tint` are left out. It is registered in the manifest,
  checked with the teacher, and its depth is checked as above.
- Eight runs: L0_bench × rgb, rgbd, rgbdt and rgbt, each with training seed 42 (the default) and 43
  (`EXTRA="training.seed=43" TAG=seed43`). Two rounds on four GPUs.

## Selection rule (fixed before the runs, applied as written after step 1)

- **rgb is judged on the 100 episodes of (a) and (g) together.** For alpha 0.2, the rgb value is the 200 episodes of
  the two existing runs together: `../build_up/L0_distract_rgb_500demos` (D1) and
  `../build_up/L0_distract_g_rgb_500demos`, 48/200 = 0.24. The two had the same training data, since the levels
  differ only in depth.
- **alpha*** is the value of {0.2, 0.3, 0.4} whose rgb is closest to 0.40. On a tie, the smaller alpha wins.
- **p\*:**
  - s_ok is the rgbd success of S5d on its normal (not see-through) episodes, (a) and (g) together (56 episodes).
  - r is the rgb of alpha*.
  - p* = (s_ok − 0.60) / (s_ok − r), rounded to the nearest 0.05 and limited to 0.2-0.8.
  - If s_ok = r, p* is undefined; the results are reported and step 2 waits.
- **Check of the see-through assumption:** the rgbd success of S5d on its see-through episodes ((a) and (g) together,
  44 episodes) is compared with rgb at alpha 0.2 (0.24, 200 episodes). If they differ by 0.15 or more with p < 0.05
  (Fisher exact), the assumption "see-through rgbd ≈ rgb" fails. Then the results are reported and step 2 does not
  start until the user decides.
- **When step 1 ends, a short report comes before step 2:** the four results, success on see-through and normal
  episodes, alpha* and the computation of p*.

## Targets and how to read the results (L0_bench, the two seeds and (a) / (g) together: 200 episodes)

| input | target | note |
|---|---|---|
| rgb | about 0.4 (0.3-0.5) | |
| rgbd | about 0.6 (0.5-0.7) | |
| rgbdt | baseline (expected about 0.65) | the simple fusion; the reference for a later fusion method |
| rgbt | baseline | how far thermal goes without depth |

- **rgb and rgbd in their ranges:** the benchmark is confirmed, and the report says that the design of the fusion
  method can start.
- **Either outside its range:** the report gives the reason (which assumption failed) and proposes the next
  adjustment, without making it.

With 50 episodes, differences under about 0.15 are within noise. Use
`python policy/summarize_runs.py compare <run A> <run B>` (Fisher exact test). With many comparisons, a p near 0.05
can be chance. Whatever the results, all runs are reported.

## Selection after step 1 (the rule above, applied as written)

Step 1 results on test sets (a) / (g), 50 episodes each (S5t was still running when the selection was made; it does
not enter the rule):

| id | run | (a) | (g) | (a) + (g) |
|---|---|---|---|---|
| A3 | `L0_distract_a03_rgb_500demos` | 0.32 (16/50) | 0.34 (17/50) | 0.33 (33/100) |
| A4 | `L0_distract_a04_rgb_500demos` | 0.48 (24/50) | 0.48 (24/50) | 0.48 (48/100) |
| S5d | `L0_distract_s50_rgbd_500demos` | 0.42 (21/50) | 0.62 (31/50) | 0.52 (52/100) |

S5d by episode type ((a) and (g) share the same 22 see-through and 28 normal episodes):
- normal: 15/28 on (a), 24/28 on (g), together **s_ok = 39/56 = 0.696** (95% CI 0.57-0.80);
- see-through: 6/22 on (a), 7/22 on (g), together 13/44 = 0.295 (95% CI 0.18-0.44).

**Check of the see-through assumption:** see-through rgbd 0.295 against rgb at alpha 0.2, 0.24 (48/200): difference
0.055, Fisher exact p = 0.45. Not 0.15 or more with p < 0.05, so the assumption "see-through rgbd ≈ rgb" holds and step
2 starts.

**alpha\*:** rgb (a) + (g) is 0.24 at alpha 0.2, 0.33 at 0.3 and 0.48 at 0.4, that is 0.16, 0.07 and 0.08 from 0.40.
**alpha\* = 0.3** (r = 0.33). Alpha 0.4 is only 0.01 further, well within noise, but the rule is applied as written.

**p\*** = (s_ok − 0.60) / (s_ok − r) = (0.696 − 0.60) / (0.696 − 0.33) = 0.096 / 0.366 = 0.263, rounded to the nearest
0.05: **p\* = 0.25** (inside 0.2-0.8). Expected rgbd on L0_bench: (1 − 0.25) × 0.696 + 0.25 × 0.33 = 0.60.

s_ok came out lower than the 0.84 of L0_distract rgbd that the plan assumed, mostly from (a) (15/28 normal episodes, against
24/28 on (g)). The rule uses the measured s_ok, so p* is 0.25 rather than the about 0.5 that s_ok = 0.84 would give.

**L0_bench** (`sim/configs/L0_bench.yaml`, meta/ep_level 30, registered in the manifest): base `L0_distract.yaml`,
`alpha: [0.3, 0.3]`, `tint: 0.0`, `glass_seethrough_p: 0.25`. Checks:
- Config loading ((a) and (g)): against L0_distract it differs only in `name`, `level`, `alpha` (None → (0.3, 0.3)) and
  `glass_seethrough_p` (0 → 0.25); every other key as listed for the step 1 levels.
- First frames against L0_distract_a03 (same alpha), training seeds 0-19 and seeds 100000-100009 of (a) and (g): 0
  mismatches. RGB, thermal, segmentation, joints and episode info are equal. Depth is equal in normal episodes; in
  see-through episodes it differs only on the glass where A3 has a dot hole, and every glass pixel holds the depth
  behind. See-through in 7 of training seeds 0-19 and 2 of test seeds 100000-100009.
- Over all seeds: see-through in 128 of the 500 training seeds and in 11 of the 50 test seeds (the same 11 in (a)
  and (g): 100000, 100006, 100015, 100018, 100024, 100031, 100033, 100034, 100038, 100046, 100047). The see-through
  draw is the same uniform number as in S5 (p 0.5), so these 11 are a subset of S5's 22.
- Teacher (`teacher_check_bench.txt`): 50/50 on training seeds and on test set (g), no distractor moved more than 2 cm.
- Depth (`glass_depth_bench.png`, `python sim/preview_glass_depth.py --levels L0_distract_a03 L0_bench --seeds 0 1 2 3 4 5 6 7`):
  see-through seeds 0, 2, 6, 7 have no missing or wrong glass pixel; seeds 1, 3, 4, 5 have 0.37 / 0.38 / 0.50 / 0.44 of
  the glass missing, the same as A3. No pixel off the glass is missing.

## Results

All numbers below are computed from the `episodes.json` files of the runs (script output kept with the analysis; 95%
intervals are Wilson intervals, p values two-sided Fisher exact tests). With 50 episodes per test set, differences
under about 0.15 are within noise, and with this many comparisons a p near 0.05 can be chance.

### Failed runs

No run failed in the end; every run has its (a) and (g) evaluation. Some attempts failed at the start of training with
CUDA out of memory, because another user's jobs on the shared GPUs took the free memory while our training was still
allocating its own (the logs end in `torch.OutOfMemoryError: CUDA out of memory ... of which 33-86 MiB is free`):
- step 1: the smoke test once (`data/runs/smoke_bench1_oom1.log`), A4 once, S5d three times. All were retried and
  finished. After the third S5d failure the runner was replaced by one with 30 retries per run (same runs, same
  settings), which took over A4 while it was running.
- step 2: no failed attempt.

Order: step 2 was started at 10:33 KST, before S5t (which does not enter the selection) had finished, at the user's
request. The selection had been made from A3, A4 and S5d before that.

### Summary

`python policy/summarize_runs.py table results/hard_scenario/benchmark/*/` is saved as `summary_table.md` (all 12
runs: success, failure types, training rollouts, validation loss, success by condition).

### Step 1

**rgb by alpha** (alpha 0.2 = the two existing build-up runs, 200 episodes):

| alpha | (a) | (g) | (a) + (g) | \|rgb − 0.40\| | failure types ((a) + (g)) | last gripper close to the target, median: all / failed episodes |
|---|---|---|---|---|---|---|
| 0.2 | 19/100 | 29/100 | 48/200 = 0.24 (0.19-0.30) | 0.16 | missed_grasp 98, wrong_height 33, dropped 10, other 8, wrong_object 2, timeout 1 | 2.5 / 3.4 cm |
| 0.3 | 16/50 | 17/50 | 33/100 = 0.33 (0.25-0.43) | 0.07 | missed_grasp 45, wrong_height 9, dropped 8, other 4, wrong_object 1 | 2.0 / 3.0 cm |
| 0.4 | 24/50 | 24/50 | 48/100 = 0.48 (0.38-0.58) | 0.08 | missed_grasp 34, dropped 9, wrong_height 5, other 4 | 1.2 / 2.6 cm |

The less transparent glass helps RGB the way intended: the gripper closes closer to the glass (median 2.5 → 2.0 → 1.2
cm) and there are fewer missed grasps. The failures stay "close but not on it" (failed episodes close 2.6-3.4 cm from
the target), not wrong objects.

**S5 (see-through in half of the episodes) by episode type** ((a) and (g) share the same 22 see-through and 28 normal
episodes):

| run | normal (a) / (g) | normal (a) + (g) | see-through (a) / (g) | see-through (a) + (g) | crossover (a) + (g) |
|---|---|---|---|---|---|
| S5d rgbd | 15/28, 24/28 | 39/56 = 0.70 (0.57-0.80) | 6/22, 7/22 | 13/44 = 0.30 (0.18-0.44) | 7/12 |
| S5t rgbdt | 17/28, 18/28 | 35/56 = 0.63 (0.49-0.74) | 13/22, 10/22 | 23/44 = 0.52 (0.38-0.66) | 1/12 |

- In S5, see-through rgbd (0.30) is at rgb level: 0.24 at alpha 0.2, difference 0.055, p = 0.45. Its failed
  see-through episodes close 3.2 cm from the glass (median), against 2.3 cm for failed normal episodes.
- rgbdt fills part of the see-through loss: 23/44 against 13/44 (+0.23, p = 0.050). On normal episodes it is not
  better (35/56 against 39/56, p = 0.55). It fails on thermal crossover (1/12, against 7/12 for rgbd).
- Selection (see "Selection after step 1"): alpha* = 0.3 (0.07 from 0.40, against 0.08 for alpha 0.4), and
  p* = (0.696 − 0.60) / (0.696 − 0.33) = 0.096 / 0.366 = 0.263 → 0.25.

### Step 2: L0_bench (alpha 0.3, see-through p 0.25)

| input | seed 42 (a) / (g) | seed 43 (a) / (g) | together (200 episodes, 95% CI) | target | met |
|---|---|---|---|---|---|
| rgb | 0.40 / 0.34 | 0.44 / 0.34 | 76/200 = **0.38** (0.32-0.45) | 0.3-0.5 | yes |
| rgbd | 0.60 / 0.64 | 0.66 / 0.60 | 125/200 = **0.625** (0.56-0.69) | 0.5-0.7 | yes |
| rgbdt | 0.58 / 0.54 | 0.70 / 0.60 | 121/200 = **0.605** (0.54-0.67) | baseline (expected about 0.65) | - |
| rgbt | 0.56 / 0.66 | 0.42 / 0.42 | 103/200 = **0.515** (0.45-0.58) | baseline | - |

**Comparisons** (`python policy/summarize_runs.py compare`, successes out of 50, Fisher p):

| pair | seed 42 (a) | seed 42 (g) | seed 43 (a) | seed 43 (g) |
|---|---|---|---|---|
| rgb vs rgbd | 20 vs 30, p 0.071 | 17 vs 32, p 0.005 | 22 vs 33, p 0.044 | 17 vs 30, p 0.016 |
| rgbd vs rgbdt | 30 vs 29, p 1 | 32 vs 27, p 0.42 | 33 vs 35, p 0.83 | 30 vs 30, p 1 |
| rgbt vs rgbdt | 28 vs 29, p 1 | 33 vs 27, p 0.31 | 21 vs 35, p 0.008 | 21 vs 30, p 0.11 |
| rgb vs rgbt | 20 vs 28, p 0.16 | 17 vs 33, p 0.003 | 22 vs 21, p 1 | 17 vs 21, p 0.54 |

Only rgb < rgbd holds in all four columns. rgbd and rgbdt are equal. rgbt vs rgbdt and rgb vs rgbt each differ in one
column only, and in opposite seeds.

**Where each input fails** (both seeds, (a) + (g); 44 see-through and 24 crossover episodes out of 200):

| input | normal | see-through | no crossover | crossover | normal, no crossover | normal + crossover | see-through, no crossover | see-through + crossover |
|---|---|---|---|---|---|---|---|---|
| rgb | 58/156 = 0.37 | 18/44 = 0.41 | 68/176 = 0.39 | 8/24 = 0.33 | 51/136 | 7/20 | 17/40 | 1/4 |
| rgbd | 123/156 = 0.79 | **2/44 = 0.05** | 107/176 = 0.61 | 18/24 = 0.75 | 106/136 | 17/20 | 1/40 | 1/4 |
| rgbdt | 111/156 = 0.71 | 10/44 = 0.23 | 111/176 = 0.63 | 10/24 = 0.42 | 101/136 | 10/20 | 10/40 | 0/4 |
| rgbt | 79/156 = 0.51 | 24/44 = 0.55 | 90/176 = 0.51 | 13/24 = 0.54 | 68/136 | 11/20 | 22/40 | 2/4 |

- **Depth that sees through the glass is worse than no depth.** On see-through episodes rgbd succeeds 2/44, against
  18/44 for rgb on the same episodes (p = 7e-5). Its failed see-through episodes close a median 6.6 cm from the glass
  (30 of 42 more than 5 cm away), against 1.7 cm for its failed normal episodes. The policy trusts the depth image,
  which shows only the table, and reaches for the wrong place.
- **The simple fusion does not pick the right sensor.** rgbdt recovers some of the see-through episodes (10/44 against
  2/44 for rgbd, p = 0.026), but far less than rgbt (24/44, p = 0.004 against rgbdt). On crossover episodes it loses
  against rgbd (10/24 against 18/24, p = 0.039), and on normal episodes it is 0.08 below rgbd (p = 0.15). It ends
  equal to rgbd overall (p = 0.76).
- rgbt is the only input that is about as good on see-through as on normal episodes (thermal does not depend on
  depth), and it is not hurt by crossover in this sample (13/24; crossover episodes are only 12% of the test seeds).

**Expected against actual rgbd.** The formula gave (1 − 0.25) × 0.696 + 0.25 × 0.33 = 0.605, and the result is
0.625. That match is a coincidence of two errors that cancel:
- normal episodes do better than s_ok: 123/156 = 0.79 against 0.70 in S5 (p = 0.20; L0_bench trains on three
  normal demos out of four instead of one out of two, with RGB at alpha 0.3);
- see-through episodes do much worse than rgb: 0.05 against 0.38 (p = 4e-6), not "≈ rgb" as in S5;
- the test sets hold 22% see-through episodes (11 of 50), not 25%.

Recombined with the L0_bench terms, 0.75 × 0.79 + 0.25 × 0.05 = 0.60. The assumption "see-through rgbd ≈ rgb" held at
p = 0.5 (S5) and fails at p = 0.25: with fewer see-through demos the policy relies more on depth.

**Seeds** (the same setup with training seed 42 and 43, 100 episodes each):

| input | seed 42 | seed 43 | difference | p |
|---|---|---|---|---|
| rgb | 37/100 | 39/100 | +0.02 | 0.88 |
| rgbd | 62/100 | 63/100 | +0.01 | 1 |
| rgbdt | 56/100 | 65/100 | +0.09 | 0.25 |
| rgbt | 61/100 | 42/100 | −0.19 | 0.011 |

rgb and rgbd are stable across seeds; rgbdt moves by 0.09; rgbt by 0.19 (seed 43 has 41 missed grasps against 26). A
single seed of rgbt would have given either 0.61 or 0.42.

## Conclusion

**The benchmark can be confirmed.** On L0_bench, rgb is 0.38 (0.32-0.45) and rgbd is 0.625 (0.56-0.69), both inside
their target ranges, both stable across the two seeds, and rgb < rgbd in every seed and test set. Its character is
not the one the plan assumed: rgbd is about 0.6 because it is very good on normal episodes (0.79) and almost always
fails on see-through ones (0.05, below rgb), not because see-through episodes fall to rgb level. For a fusion method
that is a sharper test: in 22% of the episodes depth is confidently wrong, and in 12% thermal is blind.

**Numbers for the fusion method** (rgbdt baseline 0.605 = 121/200):
- target 0.8-0.9: +0.20 to +0.30, that is 39 to 59 more successes out of 200. The baseline fails 79 episodes:
  - 35 normal episodes without crossover (101/136 succeed);
  - 10 crossover episodes with normal depth (10/20; rgbd gets 17/20);
  - 30 see-through episodes without crossover (10/40; rgbt gets 22/40);
  - 4 episodes that are both (0/4).
- If every see-through and every crossover episode succeeded and the normal ones stayed as they are, rgbdt would
  reach 165/200 = 0.825. So 0.8 needs almost all of them, and 0.9 also needs the normal episodes without crossover at
  0.85 or more (116/136, against 0.74 now and 0.78 for rgbd).
- Picking the best existing input in each group (rgbd on normal depth, rgbt on see-through) gives 147/200 = 0.735. A
  fusion that only switches to the right sensor reaches about 0.74 on this benchmark; the see-through episodes need
  more than rgbt's 0.55 to get to 0.8.

**Next steps (proposed, not run):**
1. The two fusion methods from the team (the cue-driven gate of `policy/cue-gating` and the learned attention weighting
   of `fusion/`) are being trained on L0_bench at the user's request, with the same protocol and two seeds; their
   results go to `results/hard_scenario/fusion_bench/`.
2. For the fusion comparisons, evaluate on more of the two hard episode types: only 44 see-through and 24 crossover
   episodes are in the 200. A test set with see-through in every episode and test set (f) (crossover) would measure
   them directly.
3. A third training seed of rgbt and rgbdt, since they move by 0.19 and 0.09 between seeds.
