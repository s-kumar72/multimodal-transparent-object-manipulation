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
