# Visibility-cell benchmark (level `L0_grid`): design and checks, written before any policy is trained

**Goal.** A benchmark in which each of the three sensors is sometimes the one that shows the glass and sometimes
blind to it, so that the value of a sensor and of a gate that picks the sensors can be measured per scene type.

**Why a new level** (`../fusion_bench/README.md`, "Why the work on L0_bench was stopped"). In `L0_bench` rgb shows the
glass in every episode, and the hard episodes only take depth or thermal away (see-through 21% of the test episodes,
thermal crossover 9%). Thermal could be switched off at a cost of 0.8 points, and no mix of those scene types changes
that.

**Decisions of 11/19** (by the project owner): the seven-cell design and the training ratio below, 500 demos, table
and lighting fixed as in `L0_bench`. The definition of "thermal blind" was left open and is set here by measurement
([Thermal blind](#thermal-blind-how-the-number-was-set)).

The design, the checks and the reporting rules below were fixed before any policy was trained on this level. The
first runs are the [baselines](#baselines-approved-1119-listed-before-they-start). Their results were added on 11/19
in a section of their own ([Baseline results](#baseline-results-1119)); the sections before it are as they were
written before the runs.

## Design

Every episode belongs to one **visibility cell**: which of rgb / depth / thermal show the glass. All combinations
occur except "none".

| cell | sensors that show the glass | rgb | depth | thermal | training demos (of 500) | test episodes per test set (of 420) |
|---|---|---|---|---|---|---|
| 0 | rgb, depth, thermal | shows | shows | shows | 140 | 60 |
| 1 | depth, thermal | blind | shows | shows | 70 | 60 |
| 2 | rgb, thermal | shows | blind | shows | 70 | 60 |
| 3 | rgb, depth | shows | shows | blind | 70 | 60 |
| 4 | thermal | blind | blind | shows | 50 | 60 |
| 5 | depth | blind | shows | blind | 50 | 60 |
| 6 | rgb | shows | blind | blind | 50 | 60 |

What "blind" and "shows" mean for each sensor:

| sensor | blind | shows (as in `L0_bench`'s normal episodes) | what the blind case stands for |
|---|---|---|---|
| rgb | room light at 2% of normal, with sensor noise of std 6 (of 255) on the image. The first frame's mean brightness is 4.8 of 255, against 197 in normal light. | normal light, glass at alpha 0.3 | a dark room. It is the low-light lever of L4 (dim light and the same noise) taken to the point where the image shows nothing. |
| depth | see-through: the glass pixels hold the depth of the surface behind, with no holes | holes on 40% of the glass pixels | a time-of-flight camera on clear glass (the L515 of TRansPose has a reading on 86% of the glass pixels, 95% of them more than 2 cm off; `real_data/README.md`) |
| thermal | the apparent temperature of the glass is within 0.1 °C of that of the table behind it | the glass is 5–35 °C from the room temperature | glass that has taken the temperature of the table it stands on |

- **A cell sets all three sensors.** A sensor that is not blind in the cell shows the glass in that episode, so
  the cell is the whole truth about visibility. `glass_seethrough_p` and `thermal.crossover_p` are not used.
- **Rgb blind hides the whole scene, depth blind and thermal blind hide only the glass.** In a dark episode the arm
  and the distractors are not in the rgb image either. Depth and thermal do not depend on visible light.
- **Everything else is `L0_bench`:** plain table, normal light (except in rgb-blind cells), 1–3 opaque distractors,
  the five objects, thermal `calibrated_v1`, no liquid.
- **Cell and object are dealt together.** Seed `k` gets object `k % 5` and a cell from a fixed shuffled pattern
  (`sim/config.py`, `cell_pattern`), so every 250 consecutive training seeds hold each (cell, object) pair exactly
  `weight` times, and every 35 consecutive test seeds hold each pair once.
  - Training (weights 14 : 7 : 7 : 7 : 5 : 5 : 5): 500 demos = 28 / 14 / 14 / 14 / 10 / 10 / 10 per object.
  - Test (equal weights): 420 episodes = 12 per (cell, object).
- **Test sets** (`sim/configs/manifest.json`, seeds 100000–100419):
  - `h_cells`: training conditions.
  - `i_cells_heldout`: the held-out distractor objects (`opaque_bottle`, `opaque_cup`). Table and light have a
    single value in this level, so the distractors are the only held-out factor, as in `L0_bench`'s test set (g).
- **Headline number: the macro average**, the mean of the seven per-cell success rates over both test sets (120
  episodes per cell and training seed). Every cell counts the same, whatever its share of the training demos. The
  per-cell table is always reported with it (`policy/summarize_cells.py`).

Figure: [`figure_cells.png`](figure_cells.png), the three policy inputs of one scene in each of the seven cells
(`sim/figure_cells.py`).

## Thermal blind: how the number was set

**Definition.** In a thermal-blind episode the glass temperature is chosen so that its *apparent* temperature is within
±0.1 °C (uniform) of the table's. Apparent temperature is what the camera measures: emissivity × emitted radiance +
(1 − emissivity) × reflected room radiance (`sim/sensors.py`, `matching_temp`). 0.1 °C is about the camera noise: 2.73
counts per pixel at 34.4 counts per °C is 0.08 °C.

**Why not the old crossover definition.** `thermal.crossover_gap` puts the glass within ±1 °C of the *room*
temperature. The table has its own temperature (room ± 1 °C), and the table is behind 96% of the glass pixels in the
image. So a glass at exactly room temperature still stands 0.47 °C from the table on average.

**Measurement** (`policy/sweep_thermal_contrast.py`, [`thermal_contrast_sweep.txt`](thermal_contrast_sweep.txt)): first
frame of 300 `L0_bench` test episodes, glass temperature set by hand, and the `L0_bench` segmentation network (`mix_ws`)
given the thermal image only.
- *located*: the predicted glass centre is within 4 px of the true one.
- *chance*: the same when each prediction is compared with the glass of another episode. The glass spawns in a small
  part of the 96×96 image, so a guess that ignores the frame is within 4 px in 0.28 of the frames.

| contrast (°C) | against the table (apparent): located | against the room (old definition): located | old definition: mean apparent contrast against the table (°C) |
|---|---|---|---|
| 0 | 0.233 | 0.313 | 0.47 |
| 0.05 | 0.233 | 0.310 | 0.48 |
| 0.1 | 0.250 | 0.313 | 0.48 |
| 0.2 | 0.250 | 0.337 | 0.51 |
| 0.3 | 0.273 | 0.343 | 0.54 |
| 0.5 | 0.337 | 0.380 | 0.62 |
| 0.75 | 0.400 | 0.413 | 0.76 |
| 1.0 | 0.480 | 0.453 | 0.95 |
| chance | 0.28 | 0.27–0.28 | |

- Against the table, the detector stays at chance up to 0.3 °C and rises from 0.5 °C. ±0.1 °C leaves a factor of
  three to where detection starts.
- With the old definition the detector is above chance however small the gap to the room temperature is. This is the
  leak the first draft of this level showed (thermal alone located the glass in 0.35–0.60 of the frames of the
  thermal-blind cells).
- The hand rule of the visibility gate (thermal step > 30 noise units) calls none of these frames visible.

**How often is real glass thermally blind?** (`real_data/thermal_contrast.py`,
[`real_thermal_contrast.txt`](real_thermal_contrast.txt): |median on a glass region − median on a 12 px ring around it|)

| dataset | glass | regions | median contrast | share of regions at or below the blind level |
|---|---|---|---|---|
| RGB-T Glass (thermal in °C) | panes: windows, doors, walls | 2290 in 751 frames | 0.14 °C | 0.42 below 0.1 °C (0.60 below 0.2 °C, 0.91 below 1 °C) |
| TRansPose (raw counts, not calibrated) | tabletop transparent objects | 1548 in 770 frames | 42 counts = 47 × its camera noise | 0.03 below 3 counts (0.01 below 1 count, 0.11 below 10 counts) |

- Thermal blindness is a real condition, and how common it is depends on the setting: frequent for glass panes at
  room temperature, rare for the handled tabletop objects of TRansPose (which are mostly colder than what is around
  them, median −42 counts).
- 3 counts is about 0.1 °C only if TRansPose's camera has the sim's 34 counts per °C. The sim's gain was fitted to
  that dataset's count level, not calibrated, so read the TRansPose column in noise units.
- **The cells are a controlled design, not a model of how often each case happens.** The equal test weights say
  nothing about frequency in the field.

## Implementation

All new behaviour is behind `cells` (default `None`). Existing levels draw the same random numbers as before.

- `sim/config.py`: `SceneConfig.cells`, `dark_light`, `dark_rgb_noise`; `ThermalConfig.blind_contrast`;
  `cell_pattern`; RNG stream `cell` (id 8).
- `sim/env.py`: `_cell` (cell and object of a seed), the `dark` light (normal × `dark_light`, speculars too) and its
  rgb noise, the glass temperature of a thermal-blind episode. Episode info gets `cell`, `rgb_dark` and
  `thermal_contrast` (apparent, glass − table, °C). `collect_demos.py` and `policy/eval.py` store them per episode.
- `sim/sensors.py`: `force_seethrough` in `Sensors.episode`; `apparent_contrast`, `matching_temp`.
- `sim/configs/L0_grid.yaml` (base `L0_bench.yaml`, `meta/ep_level` 31); test sets `h_cells` and `i_cells_heldout`
  in `sim/configs/manifest.json`.
- Tools: `policy/check_cells.py` (blindness per cell), `policy/sweep_thermal_contrast.py`, `policy/summarize_cells.py`
  (per-cell success and macro average), `sim/figure_cells.py`, `real_data/thermal_contrast.py`;
  `sim/teacher_check.py` got `--test_set` and `--n_test`.

**Regression check** (`sim/regression_test.py`, baselines saved before the change): the default config and `L0_bench`
both give 190/190 arrays identical.

## Checks before any training

1. **Cells as designed** (`results/data/sim_demos_L0_grid_500.zarr`, meta arrays): 500 demos from seeds 0–499, none
   missing. Per cell 140 / 70 / 70 / 70 / 50 / 50 / 50, per (cell, object) 28 / 14 / 14 / 14 / 10 / 10 / 10. In every
   cell the dark, see-through and thermal-blind flags are those of the cell. Apparent contrast against the table:
   0.000–0.100 °C in thermal-blind cells, 4.4–36.8 °C in the others.
2. **Teacher** ([`teacher_check.txt`](teacher_check.txt)): 500/500 on the training seeds (100/100 per object),
   420/420 on each test set, no distractor moved more than 2 cm. The teacher reads the simulator state, so darkness
   does not affect it.
3. **Blindness per cell, `L0_bench` network** (`policy/check_cells.py` with `mix_ws`, which never saw this level;
   [`cells_check_L0_bench_net.txt`](cells_check_L0_bench_net.txt)). First frame of the 840 test episodes, 120 per
   cell, located / chance:

   | cell | shows the glass | rgb only | depth only | thermal only | the visible sensors | all three |
   |---|---|---|---|---|---|---|
   | 0 | rgb, depth, thermal | 0.942 / 0.362 | 0.992 / 0.358 | 0.908 / 0.354 | 0.983 | 0.983 |
   | 1 | depth, thermal | **0.025 / 0.045** | 0.975 / 0.336 | 0.925 / 0.331 | 0.967 | 0.967 |
   | 2 | rgb, thermal | 0.992 / 0.366 | **0.358 / 0.298** | 0.958 / 0.362 | 1.000 | 1.000 |
   | 3 | rgb, depth | 0.942 / 0.350 | 1.000 / 0.349 | **0.242 / 0.260** | 1.000 | 1.000 |
   | 4 | thermal | **0.067 / 0.057** | **0.250 / 0.212** | 0.975 / 0.322 | 0.975 | 0.983 |
   | 5 | depth | **0.058 / 0.053** | 0.992 / 0.293 | **0.200 / 0.219** | 0.992 | 0.992 |
   | 6 | rgb | 0.958 / 0.321 | **0.208 / 0.200** | **0.200 / 0.243** | 0.958 | 0.908 |

   Bold = the sensor is blind in the cell. Every blind sensor is at chance (largest excess +0.06, cell 2 depth), every
   visible sensor locates the glass in at least 0.908 of the frames. The gate's hand rule at the true mask agrees
   with the cell: depth shows in 0.975–0.992 of the frames of the depth-visible cells and 0.000 of the others,
   thermal 0.992–1.000 and 0.000.
   - Only the first frame is used. Later frames come from the teacher, whose arm is already on its way to the glass
     and shows where it is in every sensor.
4. **Blindness per cell, network trained on this level.** The stricter test: a network that has seen the blind
   cells in training could learn a faint signal the old one ignores. Same check with the level's own network
   (`mix_ws` recipe on `L0_grid` frames of 500 training episodes plus the real datasets).
   **Criteria, written before the network finished training:**
   - (a) every blind sensor alone: located ≤ chance + 0.10 (9 cell–sensor pairs);
   - (b) every visible sensor alone: located ≥ 0.85 (12 pairs);
   - (c) the visible sensors together: located ≥ 0.90 in every cell.

   If (a) fails for thermal, `blind_contrast` is lowered and the demos are collected again; a failure of (b) or (c)
   is reported and decided with the project owner. Result: [below](#result-of-check-4).
5. **Evaluation pipeline** (not a result): a policy trained on `L0_bench` ran through `policy/eval.py` on 35 episodes
   of each test set; every episode record carries its `cell`, and `policy/summarize_cells.py` reads them.

### Result of check 4

Network: `data/grid_bench/runs/mix_ws` (20000 iterations, 47 min; finished after the criteria above were committed).
First frame of the 840 test episodes, located / chance ([`cells_check_L0_grid_net.txt`](cells_check_L0_grid_net.txt)):

| cell | shows the glass | rgb only | depth only | thermal only | the visible sensors | all three |
|---|---|---|---|---|---|---|
| 0 | rgb, depth, thermal | 0.908 / 0.363 | 0.983 / 0.359 | 0.792 / 0.344 | 0.975 | 0.975 |
| 1 | depth, thermal | **0.375 / 0.316** | 0.975 / 0.334 | 0.767 / 0.317 | 0.967 | 0.967 |
| 2 | rgb, thermal | 0.983 / 0.362 | **0.350 / 0.312** | 0.908 / 0.352 | 1.000 | 1.000 |
| 3 | rgb, depth | 0.942 / 0.351 | 1.000 / 0.348 | **0.142 / 0.111** | 1.000 | 1.000 |
| 4 | thermal | **0.383 / 0.322** | **0.283 / 0.220** | 0.933 / 0.321 | 0.933 | 0.958 |
| 5 | depth | **0.283 / 0.307** | 0.992 / 0.293 | **0.150 / 0.154** | 0.992 | 0.992 |
| 6 | rgb | 0.933 / 0.324 | **0.250 / 0.212** | **0.150 / 0.109** | 0.933 | 0.933 |

- **(a) met.** Every blind sensor is at chance for the network that was trained on the blind cells too: largest
  excess +0.063 (cell 4, depth). The level's `blind_contrast` stays 0.1 °C and the demos are kept.
- **(b) not met for 2 of the 12 pairs:** thermal alone in cells 0 and 1 (0.792 and 0.767, limit 0.85). The other ten
  pairs are 0.908–1.000.
- **(c) met:** the visible sensors together locate the glass in 0.933–1.000 of the frames of every cell.
- **What the miss in (b) is.** On the same frames the `L0_bench` network locates the glass from thermal alone in
  0.908 and 0.925 (check 3), so the thermal image carries the signal and the level's network is the weaker
  thermal-only detector. Its misses are mostly glass colder than the room (0.64–0.71 located, against 0.86–0.88 for
  warmer glass).
- **"Alone" is exact.** The mask predicted from thermal alone does not change when the rgb and depth images are
  replaced by zeros (0 of 840 frames, both networks).
- **Decision:** reported to the project owner on 11/19. Proposed: the level stays as it is, because (b) was there to
  catch a sensor that does not show the glass where it should, and check 3 shows that it does. The weakness is the
  network's and matters for the gate, which uses this network to find the glass; it belongs to the gate design. The
  baselines do not use the network.

## Reporting rules, fixed before any policy is trained

- **Evaluation:** the last checkpoint (epoch 100) on `h_cells` and `i_cells_heldout`, 420 episodes each.
- **Reported for every policy:** the per-cell table (120 episodes per cell and training seed, both test sets
  pooled; the two test sets also separately) and the macro average with its bootstrap interval.
- **Comparing two policies:** paired on the same episodes (sign test), for the macro average and per cell; with
  several training seeds, the pooled test and the seed-level one, as in `fusion_bench`.
- **Cells a policy cannot solve.** A policy without any sensor that shows the glass has nothing to go on: rgb in
  cells 1, 4, 5; rgbd in cell 4; rgbt in cell 5. Its success there is expected near 0 and is reported. If it is
  above 0.15, the cell leaks and is examined before the results are used.

## Limits

- **Rgb is a weak sensor even when it shows the glass.** The glass keeps `L0_bench`'s alpha 0.3, chosen there so that
  an rgb policy reaches about 0.4. The rgb-only cell (6) therefore has a lower ceiling than the depth-only or
  thermal-only cells.
- **Thermal "shows" means a large gap** (5–35 °C, a hot or cold drink). Real empty glass is nearer the middle:
  TRansPose objects are a median 47 noise units from their surroundings, which the sim's scale puts at roughly 1.2 °C.
  A level with such faint thermal episodes is not part of this benchmark.
- **Table and light do not vary** (decision of 11/19: one change at a time).
- **The 4 px criterion has a high chance level** (0.20–0.37 whenever the network predicts a mask), because the
  glass spawns within about 12 × 30 cm. All blindness statements are relative to that chance level.

## Baselines (approved 11/19, listed before they start)

Plain Diffusion Policy, one run per sensor set, to see what the benchmark does to policies that have no gate.

| run | sensors | seed |
|---|---|---|
| `L0_grid_rgbdt_500demos` | rgb, depth, thermal | 42 |
| `L0_grid_rgbd_500demos` | rgb, depth | 42 |
| `L0_grid_rgbt_500demos` | rgb, thermal | 42 |
| `L0_grid_rgb_500demos` | rgb | 42 |

- **Settings:** those of the `L0_bench` baselines (`scripts/run_experiment.sh L0_grid <variant> 500`, default config,
  101 epochs), except the rollouts during training, which use `h_cells` with 35 episodes (5 per cell, monitoring
  only), and `TOPK=0` (only the last checkpoint is kept, the one that is evaluated).
- **Final evaluation:** as in the reporting rules, `TEST_SETS="h_cells i_cells_heldout" NEVAL=420` (new knobs of
  `scripts/run_experiment.sh`; its defaults are unchanged).
- **What to expect, written before the runs (not a criterion):**
  - near 0 in the cells a policy cannot solve: rgb in cells 1, 4, 5; rgbd in cell 4; rgbt in cell 5;
  - rgbdt has a sensor that shows the glass in every cell, so any cell where it is far below the others shows what a
    policy without a gate loses to its blind sensors.
- **Not part of this step:** seeds 43 and 44, and the gate for three sensors. The gate so far decides depth and
  thermal only; it needs a third flag for rgb and the level's segmentation network (check 4). Its design is written up
  and approved before it is trained.

## Baseline results (11/19)

The four runs finished (101 epochs, last checkpoint, 420 episodes on each of `h_cells` and `i_cells_heldout`; seed 42
only). Success per cell, both test sets pooled (120 episodes per cell); `baselines_cells.txt` has the tables per test
set and the paired comparisons (`policy/summarize_cells.py`, option `--versus` for the pairs).

| cell | sensors that show the glass | rgbdt | rgbd | rgbt | rgb |
|---|---|---|---|---|---|
| 0 | rgb+depth+thermal | 0.625 | 0.600 | 0.217 | 0.025 |
| 1 | depth+thermal | 0.575 | 0.750 | 0.167 | 0.033 |
| 2 | rgb+thermal | 0.083 | 0.025 | 0.242 | 0.033 |
| 3 | rgb+depth | 0.550 | 0.583 | 0.042 | 0.042 |
| 4 | thermal | 0.017 | 0.025 | 0.158 | 0.017 |
| 5 | depth | 0.575 | 0.625 | 0.050 | 0.025 |
| 6 | rgb | 0.050 | 0.017 | 0.042 | 0.025 |
| | **macro average** | **0.354** | **0.375** | **0.131** | **0.029** |
| | bootstrap 95% | 0.325–0.381 | 0.350–0.400 | 0.111–0.154 | 0.018–0.040 |
| | `h_cells` / `i_cells_heldout` | 0.383 / 0.324 | 0.398 / 0.352 | 0.136 / 0.126 | 0.031 / 0.026 |

**Cells a policy cannot solve** (reporting rule): rgb in cells 1, 4, 5: 0.033, 0.017, 0.025; rgbd in cell 4: 0.025;
rgbt in cell 5: 0.050. All far below 0.15: no cell leaks.

What the table shows:

- **A floor of 0.02–0.05.** Wherever a policy has nothing to go on it still succeeds in 2–6 of 120 episodes (the
  glass spawns in a small area). In those cells half of the episodes or more end with the gripper closing more than
  8 cm from the glass (`other`, 47–79 of 120), which does not happen once in the cells where depth shows the glass to
  a policy that has depth.
- **Depth works as designed.** rgbd reaches 0.583–0.750 in the four cells where depth shows the glass (307/480 =
  0.640) and the floor in the other three (8/360 = 0.022).
- **Thermal works, at a low level.** rgbt reaches 0.158–0.242 in the four cells where thermal shows the glass
  (94/480 = 0.196) and 0.042–0.050 in the other three (16/360 = 0.044). Where thermal shows the glass, the policy
  goes to it but does not grasp it: 62–71 of 120 episodes end with a close within 8 cm and no lift (`missed_grasp`),
  6–11 as `other`.
- **Rgb is not used by any policy.** The rgb policy is at the floor in the four cells where rgb shows the glass
  (15/480 = 0.031) as in the three dark ones (9/360 = 0.025), with the same mix of outcomes; 13 of its 24 successes
  are the jar. The rgb-only cell (6) is at the floor for every policy (0.017–0.050), and neither rgbd nor rgbt gains
  anything from rgb in the cells where their other sensor is blind (rgbd in cell 2: 0.025; rgbt in cell 3: 0.042).
  The first point of [Limits](#limits) expected about 0.4 for an rgb policy where rgb shows the glass, as the plain
  rgb policy reaches on `L0_bench` (0.389 on two seeds, 800 episodes, `../fusion_bench/README.md`). That did not
  hold.
- **rgbdt behaves as a depth policy and loses what thermal alone gives.** It equals rgbd over all episodes (0.354
  against 0.375; same episodes: 64 only rgbdt, 82 only rgbd, sign test p 0.16). In the two cells where thermal shows
  the glass and depth does not, it is below rgbt: cell 2, 0.083 against 0.242 (5 / 24, p 0.0006); cell 4, 0.017
  against 0.158 (1 / 18, p 8e-5). In cell 1 (rgb dark) it is below rgbd, 0.575 against 0.750 (4 / 25, p 0.0001).
- **Held-out distractors cost a few points** for the policies that work (rgbdt 0.383 to 0.324, rgbd 0.398 to 0.352).
- **Arithmetic on the table, not a run:** taking rgbd in the cells where depth shows the glass and rgbt in the others
  gives (0.600 + 0.750 + 0.583 + 0.625 + 0.242 + 0.158 + 0.042) / 7 = 0.429, against 0.354 for rgbdt. That is what
  choosing between the two plain policies by the true cell would reach.

What this says about the benchmark:

- **The depth and thermal axes do what they were built for**: a policy succeeds where its sensor shows the glass and
  is at the floor where it does not, no cell leaks, and the policy with all three sensors is not the best one in the
  cells where only thermal helps. That is the gap a gate is meant to close.
- **The rgb axis does not work for plain Diffusion Policy.** "Rgb shows the glass" was checked with the segmentation
  network (checks 3 and 4: rgb alone locates it in 0.91–0.98 of the first frames), not with a policy. No baseline
  solves the rgb-only cell, so the macro average of these baselines cannot exceed 6/7, and the rgb baseline says
  nothing about rgb.
- **Why rgb fails here and not on `L0_bench` is not established.** The glass looks the same in rgb as there (alpha
  0.3, normal light). Two differences in the training data, neither tested: 170 of the 500 demos are dark (the rgb
  image carries nothing, 34%), and 330 demos with a visible glass remain instead of 500.
- **Thermal's ceiling is low** (0.16–0.24 where it is the sensor that shows the glass), so cells 2 and 4 can add
  little to the macro average as long as the policy cannot grasp from thermal.
- **One seed.** The cell rates have a 95% interval of about ±0.09 at 0.5; the statements above rest on differences
  far larger than that, except the ranking of rgbdt and rgbd.

Nothing in the level, the test sets or the reporting rules was changed after these results. Proposed next, none of
it started: (1) one diagnostic run, the plain rgb policy trained on the 330 demos of the cells where rgb shows the
glass, to tell the two candidate causes apart; (2) seeds 43 and 44 of the baselines; (3) the gate for three sensors.
Whether the level's rgb cells are changed is the project owner's decision.

## Repair round: the rgb axis and the thermal level (plan of 11/20, written before the runs)

**Why.** The aim of the project is a method on top of the plain rgbdt policy that makes it at least as good as every
other policy in nearly every cell. The baselines above leave little to show that with: no policy uses rgb (the
rgb-only cell is at the floor for all four), and thermal alone reaches 0.16-0.24. A policy that matched the best
baseline of every cell would average 0.43 against 0.375 for rgbd, with the whole gain in two cells whose ceiling is
0.24. Two more facts from the baseline records (no new run):

- In the cells where thermal shows the glass, the success of rgbt does not depend on the size of the thermal contrast
  (0.15-0.33 in every bin from 5 to 35 degC, with or without depth) or on its sign (hot 55 / 244, cold 39 / 236).
  One object carries it: jar 36 / 96, the other four 11-17 / 96.
- The same kinds of scene gave more on `L0_bench` (other test episodes, two seeds): rgb 0.39 (here 0.03 where rgb
  shows the glass), rgbt 0.50 on see-through episodes (here 0.24 in cell 2), rgbd 0.80 on normal episodes (here
  0.58-0.75 where depth shows the glass). What changed is the training set: every sensor shows the glass in 330 of
  the 500 demos, and 170 demos have a dark colour image.

**Runs** (plain Diffusion Policy, seed 42, settings and final evaluation of the baselines: level `L0_grid`, the same
840 test episodes; the level and the test sets are not changed):

| run | sensors | demos | question |
|---|---|---|---|
| `L0_grid_rgb_330demos_rgbshown` | rgb | the 330 demos of cells 0, 2, 3, 6 of the 500-demo set (`policy/subset_demos.py`) | do the dark demos keep the policy from learning rgb? |
| `L0_grid_rgb_1000demos` | rgb | 1000 (seeds 0-999, same weights; rgb shows the glass in 660) | does more data repair the rgb axis? |
| `L0_grid_rgbt_1000demos` | rgb, thermal | 1000 | does more data raise the cells where only thermal can help (2 and 4)? |
| `L0_grid_rgbt_500demos_cell4` | rgb, thermal | 500 demos of cell 4 only (dark, see-through, thermal shows the glass; `collect_demos.py --cells 4`) | what can a policy do with thermal alone when all its demos are of that kind? |

**Read-out, fixed now.** Thresholds on the pooled success over both test sets:

| quantity | now | repaired | not repaired |
|---|---|---|---|
| rgb policy in the four cells where rgb shows the glass (0, 2, 3, 6; 480 episodes) | 0.031 | >= 0.30 | < 0.15 |
| rgbt policy in cells 2 and 4 (240 episodes); for the cell-4 run, cell 4 only (120 episodes; its demos have no lit scene, so cell 2 is outside its training data) | 0.200; cell 4 0.158 | >= 0.40 | < 0.30 |

Between the two limits the result is called partial. 0.15 is the leak limit of the reporting rules (below it a policy
is not told apart from one that cannot see the glass); 0.30 is short of what rgb reached on `L0_bench` (0.39); 0.40
is twice what thermal gives now. The cells a policy cannot solve are reported as before and must stay below 0.15.

**What follows from each outcome** (also fixed now; the follow-up runs themselves are reported before they start):

- rgb with 1000 demos repaired: the training set of the benchmark becomes the 1000 demos, and rgbd and rgbdt are
  trained on them as well (rgbt is already in this round).
- rgb with 1000 demos not repaired, rgb on the 330 lit demos repaired: the dark demos are the obstacle. Rgb blind is
  then redefined to hide only the glass, as depth blind and thermal blind do, in a new level; `L0_grid` stays.
- neither repaired: neither explanation holds; reported, nothing changed.
- thermal: rgbt with 1000 demos repaired means more data is enough. If not, the cell-4 run tells which: repaired
  there means thermal alone can carry a grasp and the mixed training set is the obstacle; not repaired there means
  the limit is the thermal image as the policy gets it, and changing that is proposed, not done.

One seed per run. Nothing here changes `L0_grid`, its test sets or the baseline results above.

### First three results and the follow-up level `L0_grid_clear` (11/20 04:10, written before its runs)

Three of the four runs are in (840 test episodes each; the full tables come with the fourth):

| run | read-out | result |
|---|---|---|
| `L0_grid_rgb_330demos_rgbshown` | cells where rgb shows the glass | 162 / 480 = 0.338 (baseline 0.031): **repaired**; dark cells 7 / 360 |
| `L0_grid_rgb_1000demos` | cells where rgb shows the glass | 7 / 480 = 0.015: **not repaired**; dark cells 5 / 360 |
| `L0_grid_rgbt_500demos_cell4` | cell 4 | 69 / 120 = 0.575 (baseline 0.158): **repaired**; cell 1, the same case for this policy, 67 / 120 |

So the rule above applies: the dark demos are the obstacle for rgb, and more demos do not help (660 lit demos with
340 dark ones give nothing, 330 lit demos alone give 0.34). Rgb blind is redefined in a new level; `L0_grid` stays.

**Level `L0_grid_clear`** (`sim/configs/L0_grid_clear.yaml`, level id 32; `SceneConfig.rgb_blind: clear`): `L0_grid`
with one change. An rgb-blind episode has normal light, and the glass is clear to the colour camera (alpha 0.001,
below one grey level). The colour image shows the table, the arm and the distractors but not the glass, so rgb blind
hides only the glass, as depth blind and thermal blind do. Test sets `h_cells` and `i_cells_heldout` as before (the
same seeds and cells).

**Check** (`sim/check_clear.py`, [`clear_check.txt`](clear_check.txt); 250 training seeds and 3 teacher episodes per
cell):

- In the rgb-blind cells the camera frame with the glass and the frame rendered without the transparent objects
  differ in 0 pixels, on the first frame and on every frame of the teacher episodes (in the other cells in 220-330
  pixels of the 240 x 320 frame, by up to 190 grey levels). Blindness is exact; there is no chance level to compare
  with.
- First-frame depth and thermal equal those of `L0_grid` for all 250 seeds, and in the cells where rgb shows the
  glass the whole teacher episode is the same. In the 500 demos (teacher 500 / 500) the episodes of those cells are
  identical to `L0_grid`'s, the actions of all 500 are, and the rgb-blind episodes have a mean brightness of 197
  (4.8 in `L0_grid`).
- The existing levels are unchanged: `sim/regression_test.py` on the default config, `L0_bench` and `L0_grid`,
  190 / 190 arrays each.

**Runs:** the four baselines again, on this level: `L0_grid_clear_{rgb,rgbd,rgbt,rgbdt}_500demos`, seed 42, settings
and final evaluation of the `L0_grid` baselines.

**Read-out, fixed now:**

- rgb axis: the rgb policy over the cells where rgb shows the glass (0, 2, 3, 6), >= 0.30 repaired, < 0.15 not
  repaired, as above. If it is not repaired, demos without the glass in the colour image keep a plain policy from
  learning rgb however the glass is hidden; nothing further is changed and the choice of benchmark goes back to the
  user with both tables.
- Leak: rgb in cells 1, 4, 5, rgbd in cell 4 and rgbt in cell 5 stay below 0.15.
- Thermal: rgbt and rgbdt in cell 4, against 0.40 and 0.30 as above and against 0.575, what the cell-4 run reached.
- Reported without a threshold: the best baseline of every cell, and the mean of those against the best single
  baseline (the room a method has on this level).

### Results of the repair round and of `L0_grid_clear` (11/20; seed 42, 840 test episodes per run)

Per-cell tables of every run and the paired comparisons: [`repair_cells.txt`](repair_cells.txt).

**Repair round** (level `L0_grid`, other training sets):

| run | read-out | result | baseline | verdict by the limits above |
|---|---|---|---|---|
| rgb, the 330 lit demos | cells where rgb shows the glass (480) | 162 = 0.338 | 0.031 | repaired |
| rgb, 1000 demos | the same cells | 7 = 0.015 | 0.031 | not repaired |
| rgbt, 500 demos of cell 4 | cell 4 (120) | 69 = 0.575 | 0.158 | repaired |
| rgbt, 1000 demos | cells 2 and 4 (240) | 79 = 0.329 | 0.200 | partial |

- The cells a policy cannot solve stay at the floor in all four runs (0.008-0.050).
- The cell-4 policy is as good in cell 1 (0.558), the same case for a policy without depth, and solves all five
  objects (0.42-0.83; the baseline's successes were mostly the jar).
- With 1000 demos rgbt gains in the cells where thermal shows the glass (cell 4: 0.342 against 0.158, 28 / 6 on the
  same episodes, p 0.0002) and still does not use rgb (cells 3 and 6: 0.037).

**Baselines on `L0_grid_clear`** (500 demos), next to those on `L0_grid`:

| cell | sensors that show the glass | rgbdt | rgbd | rgbt | rgb | | rgbdt on `L0_grid` | rgbd | rgbt | rgb |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | rgb, depth, thermal | 0.592 | 0.583 | 0.167 | 0.050 | | 0.625 | 0.600 | 0.217 | 0.025 |
| 1 | depth, thermal | 0.633 | 0.700 | 0.075 | 0.033 | | 0.575 | 0.750 | 0.167 | 0.033 |
| 2 | rgb, thermal | 0.108 | 0.050 | 0.117 | 0.058 | | 0.083 | 0.025 | 0.242 | 0.033 |
| 3 | rgb, depth | 0.650 | 0.650 | 0.117 | 0.100 | | 0.550 | 0.583 | 0.042 | 0.042 |
| 4 | thermal | 0.058 | 0.042 | 0.017 | 0.042 | | 0.017 | 0.025 | 0.158 | 0.017 |
| 5 | depth | 0.625 | 0.633 | 0.033 | 0.033 | | 0.575 | 0.625 | 0.050 | 0.025 |
| 6 | rgb | 0.083 | 0.092 | 0.183 | 0.075 | | 0.050 | 0.017 | 0.042 | 0.025 |
| | macro average | 0.393 | 0.393 | 0.101 | 0.056 | | 0.354 | 0.375 | 0.131 | 0.029 |

Read-out of `L0_grid_clear`, by the limits fixed before its runs:

- **rgb axis: not repaired.** The rgb policy succeeds in 34 / 480 = 0.071 of the episodes of the cells where rgb
  shows the glass (0.036 in the others). It comes near the glass more often there (closes within 8 cm in 51% of the
  episodes, against 29% in the rgb-blind cells) but does not grasp it.
- **Leak: none.** rgb in cells 1, 4, 5: 0.033, 0.042, 0.033; rgbd in cell 4: 0.042; rgbt in cell 5: 0.033.
- **Thermal: not repaired.** Cell 4 is 0.017 for rgbt and 0.058 for rgbdt, against 0.575 for the cell-4 policy (which
  was trained and tested on `L0_grid`, where the rgb-blind image is dark).
- **Best baseline of every cell:** mean 0.419 on `L0_grid_clear` and 0.433 on `L0_grid`, against the best single
  baseline 0.393 and 0.375. With the two single-sensor reference policies of the repair round (rgb on the lit demos,
  rgbt on the cell-4 demos) the per-cell best on `L0_grid` is 0.625, 0.750, 0.367, 0.583, 0.575, 0.625, 0.308: mean
  0.548.

**What the twelve runs show.**

1. A plain policy uses one sensor. With depth it is a depth policy: rgbd and rgbdt reach 0.58-0.75 where depth shows
   the glass and 0.02-0.11 elsewhere, on both levels, although rgb or thermal show the glass in those cells. rgbdt
   is not above rgbd on either level (0.354 / 0.375, 0.393 / 0.393).
2. Without depth it leans on one of the others, weakly, and which one depends on the level: thermal on `L0_grid`
   (0.16-0.24 where thermal shows the glass, rgb unused), rgb on `L0_grid_clear` (0.12-0.18 where rgb shows the
   glass, thermal-only cell at the floor).
3. Each weak sensor can carry the task when all demos are of the kind where it shows the glass (rgb 0.31-0.37,
   thermal 0.56-0.58), and not when such demos are mixed with demos where it does not: not with twice the demos (rgb
   0.015; thermal 0.33), and for rgb not with the other way of hiding the glass either (0.071).
4. So the low rgb and thermal numbers of the baselines are not defects of the scenes; they are what a plain policy
   does with demos of mixed visibility. The cause is not established.

**Decisions of the user (11/20), which change what follows:**

- A method is trained and evaluated on sensor images and joint values only (and the demo actions in training).
  Nothing that only the simulator knows is used, in training either: no cell, no visibility flag, no glass mask and no
  network trained on such masks. The gated policies of `fusion_bench` used a segmentation network trained on
  simulator masks; they are dropped as methods and stay as a reference for what that information gives.
- The two single-sensor policies above were trained on demos chosen by cell, so they are reference levels, not
  baselines.
- All-or-nothing visibility is a limit of both levels; intermediate visibility is to be added.

**Open, nothing started:** which of the two levels becomes the benchmark (the numbers do not separate them; the
clear rgb-blind case cannot be told from image brightness); a higher rgb "shows" level, so that the single-sensor
reference of rgb is in the range of depth and thermal (0.34 now; alpha 0.3 was set on `L0_bench`, where rgb always
showed the glass); intermediate visibility levels set by measurement, first as test sets for the existing
checkpoints; seeds 43 and 44; a baseline with random sensor dropout.

### The "rgb shows" level: which alpha (plan of 11/20 09:20, written before the runs)

**Why.** A policy with rgb alone, trained on the demos where rgb shows the glass, reaches 0.338; depth gives
0.58-0.75 (rgbd where depth shows the glass) and thermal alone 0.575. Alpha 0.3 was set on `L0_bench` so that rgb
would reach only about 0.4, because rgb showed the glass in every episode there. In the cell design the need for the
other sensors comes from the cells, so rgb does not have to be kept weak. Decision of the user (11/20): raise it.

**Runs:** `L0_grid_clear_a04_rgb_330demos_rgbshown` and `L0_grid_clear_a05_rgb_330demos_rgbshown`: the rgb policy on
the 330 demos of cells 0, 2, 3, 6, with the glass at alpha 0.4 and 0.5 (levels `L0_grid_clear_a04` / `_a05`, ids 33 /
34: `L0_grid_clear` with another `alpha`). Seed 42, settings and final evaluation of the baselines, on the test sets
of the level. The demos are the episodes of the alpha-0.3 lit set (same seeds, actions, depth and thermal; 0.5% of
the colour pixels differ).

**Read-out and rule, fixed now.** The reference is the success over the four cells where rgb shows the glass (480
episodes); 0.338 at alpha 0.3. Chosen: the alpha among 0.3, 0.4 and 0.5 whose reference is closest to 0.60, if that
reference is within 0.50-0.70. If none is, nothing is chosen and a further value is proposed. The rgb-blind cells
(clear glass, the same at every alpha) are reported and must stay below 0.15.

### Result: the "rgb shows" level, and the first measurements for a faint level (11/20 10:36)

Tables and paired comparisons: [`alpha_cells.txt`](alpha_cells.txt). The rgb policy on the 330 lit demos, seed 42:

| alpha of the glass | cells where rgb shows the glass (480) | rgb-blind cells (360) | jar / the other four objects |
|---|---|---|---|
| 0.3 | 162 = 0.338 | 0.019 | 0.70 / 0.19-0.29 |
| 0.4 | 190 = 0.396 | 0.056 | 0.78 / 0.20-0.34 |
| 0.5 | 208 = 0.433 | 0.036 | 0.83 / 0.22-0.44 |

- **By the rule, nothing is chosen:** no reference is within 0.50-0.70. A less transparent glass helps (0.4 against
  0.3: 69 / 41 on the same episodes, p 0.010; 0.5 against 0.3: 91 / 45, p 0.0001; 0.5 against 0.4: 67 / 49, p 0.11),
  but by 0.04-0.06 per step of 0.1, and most failures stay near the glass (closed within 8 cm without lifting:
  180-219 of 480; dropped 38-60).
- The rgb-blind cells stay at the floor.
- The references of the three sensors were not measured alike: thermal alone (0.575) had 500 demos of one cell, rgb
  330 demos. With 500 lit demos rgb gave 0.389 at alpha 0.3 on `L0_bench` (800 episodes) and 0.48 at alpha 0.4 on
  `L0_distract_a04` (100 episodes).

**Faint level: mechanism and a first measurement.** A cell flag can now be 2, "faint" (`sim/config.py`:
`faint_alpha`, `faint_holes`, `thermal.faint_contrast`; off unless a level sets it; regression 190 / 190 on the
default config, `L0_bench` and `L0_grid`; on 40 episodes the hole share on the glass is 0.39 / 0.20 / 0.00 for shows /
faint at 0.2 / blind). `policy/sweep_visibility.py` ([`visibility_sweep.txt`](visibility_sweep.txt), first frame of
300 test episodes, one sensor's strength set by hand, both segmentation networks given that sensor only): the
strength at which the glass is located half-way between chance and the rate at full strength is

| sensor | blind | half-way for the segmentation network | shows |
|---|---|---|---|
| rgb (alpha) | 0.001 | 0.10 (located 0.06 at 0.02, 0.91 at 0.15, 0.97 from 0.3 on) | 0.3 |
| depth (share of glass pixels that are holes) | 0 | 0.20 (chance up to 0.10, 0.97 at 0.30) | 0.40 |
| thermal (degC from the table) | 0.1 | 1.7 and 2.7 for the two networks (0.84-0.94 at 5) | 5-35 |

A segmentation network and a policy do not agree on rgb: the network locates the glass in 0.95 of the frames at alpha
0.2, a policy reaches 0.34 at alpha 0.3 and 0.02 at alpha 0.05-0.1 (`L0_faint`). A faint level set by the network
would leave a policy nothing in rgb, so the faint levels are better set on policies. Not done.

**Open, nothing started:** the alpha of "rgb shows" (the rule chose none); how the references are measured, alike for
the three sensors; the faint level of each sensor; the cells and sizes of the training and test sets.

### Final level: decisions of the user and how the faint levels are set (11/20 16:00, written before any run)

Decisions of the user (11/20): "rgb shows" is alpha 0.5; the benchmark has eleven cells, the seven above, three in
which a single sensor shows the glass and only faintly (rgb, depth, thermal), and one in which all three show it
faintly; 700 training demos (the 500 as before and 50 per new cell), 60 test episodes per cell and test set.

**Faint levels.** No new training: each single-sensor reference policy that exists is evaluated in its single-sensor
cell with the cue weakened, on the same 120 test episodes (seeds 100000+, no held-out factor) per value:

| sensor | policy (trained at full strength) | values | full strength |
|---|---|---|---|
| depth | `L0_grid_clear_rgbd_500demos`, rgb and thermal blind | 10%, 20%, 30% of the glass pixels are holes | 40% |
| thermal | `L0_grid_rgbt_500demos_cell4`, rgb (dark) and depth blind | 1, 2, 3, 5 degC from the table (+-10%) | 5-35 degC |
| rgb | `L0_grid_clear_a05_rgb_330demos_rgbshown`, depth and thermal blind | alpha 0.2, 0.3, 0.4 | 0.5 |

**Rule, fixed now:** the faint level of a sensor is the value whose success is closest to half of the success at
full strength on the same episodes. The policies have never seen a weakened cue, so this measures how much of the cue
a policy trained on the full one can still use; the level's own baselines and references are measured afterwards.

## Final level `L0_vis` (11/20)

**Faint levels** ([`faint_calibration.txt`](faint_calibration.txt); rule and runs as fixed above):

| sensor | cue weakened to | success of the reference policy | full strength | half of it | chosen |
|---|---|---|---|---|---|
| depth (share of glass pixels that are holes) | 10% / 20% / 30% | 0.050 / 0.275 / 0.483 | 0.600 at 40% | 0.300 | **20%** |
| thermal (degC from the table) | 1 / 2 / 3 / 5 | 0.067 / 0.233 / 0.317 / 0.425 | 0.575 at 5-35 | 0.288 | **3 (2.7-3.3)** |
| rgb (alpha) | 0.2 / 0.3 / 0.4 | 0.183 / 0.333 / 0.500 | 0.450 at 0.5 | 0.225 | **0.2** |

Success falls smoothly as each cue weakens. For depth and thermal the chosen values are near those at which the
segmentation networks locate the glass half-way (20%, 1.7-2.7 degC); for rgb they are not (alpha 0.10 there).

**Definition** (`sim/configs/L0_vis.yaml`, level id 35): `L0_grid_clear` with the glass at alpha 0.5 where rgb shows
it, and four more cells.

| sensor | shows | faint | blind |
|---|---|---|---|
| rgb | glass at alpha 0.5 | alpha 0.2 | clear glass, not in the colour image (alpha 0.001) |
| depth | 40% of the glass pixels are holes | 20% | see-through, no holes |
| thermal | 5-35 degC from the room | 2.7-3.3 degC from the table behind it, either sign | within 0.1 degC of the table |

| cell | rgb | depth | thermal | training demos (of 700) | test episodes per test set (of 660) |
|---|---|---|---|---|---|
| 0 | shows | shows | shows | 140 | 60 |
| 1 | blind | shows | shows | 70 | 60 |
| 2 | shows | blind | shows | 70 | 60 |
| 3 | shows | shows | blind | 70 | 60 |
| 4 | blind | blind | shows | 50 | 60 |
| 5 | blind | shows | blind | 50 | 60 |
| 6 | shows | blind | blind | 50 | 60 |
| 7 | faint | blind | blind | 50 | 60 |
| 8 | blind | faint | blind | 50 | 60 |
| 9 | blind | blind | faint | 50 | 60 |
| 10 | faint | faint | faint | 50 | 60 |

- **Training and test** have the same cells and differ in proportions and seeds: the training set keeps the mix of
  `L0_grid` for cells 0-6 and adds 50 demos per new cell (seeds 0-699, every (cell, object) pair 2 x weight times);
  the test sets `j_cells11` and `k_cells11_heldout` (held-out background, lighting and distractor pools, as
  `i_cells_heldout`) hold every (cell, object) pair 12 times, seeds 100000-100659.
- **Protocol** (decision of the user, 11/20): a method is trained on the demos (sensor images, joint values, actions)
  and acts on sensor images and joint values. Cells, visibility flags, glass masks and anything else only the
  simulator knows are for the report, not for a method, in training either.
- **Checks:** teacher 700 / 700 on the training seeds and 660 / 660 on each test set
  ([`teacher_check_L0_vis.txt`](teacher_check_L0_vis.txt)); in the demos every cell has the alpha, thermal contrast
  and see-through flag of its definition (alpha 0.5 / 0.2 / 0.001; contrast 4.7-36.8, 2.71-3.29 or at most 0.10
  degC); existing levels unchanged (regression 190 / 190 on the default config, `L0_bench` and `L0_grid`).

**Reporting rules, fixed before any policy is trained on the level:**

- Evaluation: the last checkpoint on `j_cells11` and `k_cells11_heldout`, 660 episodes each.
- For every policy: the per-cell table (120 episodes per cell and seed), the macro average over the eleven cells
  (the headline number) with its bootstrap interval, and the averages over cells 0-6 and over cells 7-10.
- Two policies are compared on the same episodes (sign test), per cell and for the macro average.
- Cells a policy cannot solve, expected near 0 and examined if above 0.15: rgb in cells 1, 4, 5, 8, 9; rgbd in cells
  4 and 9; rgbt in cells 5 and 8.

**Baselines, listed before they start:** plain Diffusion Policy `L0_vis_{rgbdt,rgbd,rgbt,rgb}_700demos`, seed 42,
default config, 101 epochs, rollouts during training on `j_cells11` (55 episodes, monitoring only), `TOPK=0`. Seeds 43
and 44, the single-sensor references and other baselines are not part of this step.

### Baselines on `L0_vis` (11/20 22:05; seed 42, 1320 test episodes per run, 120 per cell)

Tables per test set and paired comparisons: [`vis_cells.txt`](vis_cells.txt).

| cell | rgb / depth / thermal | rgbdt | rgbd | rgbt | rgb |
|---|---|---|---|---|---|
| 0 | shows / shows / shows | 0.675 | 0.633 | 0.167 | 0.100 |
| 1 | blind / shows / shows | 0.567 | 0.633 | 0.025 | 0.025 |
| 2 | shows / blind / shows | 0.092 | 0.092 | 0.192 | 0.092 |
| 3 | shows / shows / blind | 0.633 | 0.600 | 0.225 | 0.100 |
| 4 | blind / blind / shows | 0.050 | 0.017 | 0.058 | 0.017 |
| 5 | blind / shows / blind | 0.492 | 0.550 | 0.075 | 0.025 |
| 6 | shows / blind / blind | 0.100 | 0.108 | 0.183 | 0.075 |
| 7 | faint / blind / blind | 0.058 | 0.050 | 0.108 | 0.050 |
| 8 | blind / faint / blind | 0.375 | 0.442 | 0.033 | 0.042 |
| 9 | blind / blind / faint | 0.067 | 0.058 | 0.050 | 0.042 |
| 10 | faint / faint / faint | 0.350 | 0.292 | 0.117 | 0.067 |
| | **macro average, eleven cells** | **0.314** | **0.316** | **0.112** | **0.058** |
| | cells 0-6 | 0.373 | 0.376 | 0.132 | 0.062 |
| | cells 7-10 | 0.212 | 0.210 | 0.077 | 0.050 |
| | `j_cells11` / `k_cells11_heldout` | 0.329 / 0.300 | 0.330 / 0.302 | 0.111 / 0.114 | 0.064 / 0.052 |

- **No leak.** Cells a policy cannot solve: rgb 0.017-0.042 (cells 1, 4, 5, 8, 9), rgbd 0.017 and 0.058 (4, 9), rgbt
  0.075 and 0.033 (5, 8); all below 0.15.
- **As on the earlier levels, a plain policy is a depth policy.** rgbdt and rgbd reach 0.49-0.68 where depth shows
  the glass, 0.38-0.44 where it shows it faintly (0.29-0.35 when all three are faint) and at most 0.11 in the five
  cells without depth, although rgb or thermal show the glass in four of them. rgbdt is not above rgbd (0.314 against
  0.316).
- Without depth, rgbt leans on rgb, weakly (0.17-0.23 where rgb shows the glass, 0.11 where faintly), and does not
  use thermal (cells 4 and 9 at the floor). rgb alone stays at 0.02-0.10 even with the glass at alpha 0.5.
- The faint cells are graded as intended for depth: 0.49-0.68 (shows), 0.38-0.44 (faint), 0.02-0.11 (blind).
- Best baseline of every cell: mean 0.354, against 0.316 for the best single baseline.

One seed. The single-sensor references of this level, seeds 43 and 44 and further baselines are not run.

## Baseline programme on `L0_vis` (plan of 11/21, written before its runs)

Decision of the user (11/21): before any method is applied, the benchmark gets reference levels and stronger
baselines. Five of the eleven cells are at the floor for every plain baseline; the programme asks whether those cells
can be solved at all on this level and whether known techniques fill them. Scenes, cells and test sets are not
changed by it.

| stage | runs | question |
|---|---|---|
| 1 | single-sensor reference policies (rgb, depth, thermal), each trained on the demos where its sensor shows the glass; they use the cells | what can be reached in each cell |
| 2 | the four plain baselines, seeds 43 and 44 | how much the baseline numbers move with the seed |
| 3 | rgbdt with random sensor dropout, three seeds | does the usual remedy fill the floor |
| 4 | single-sensor policies trained on all demos (depth, thermal; rgb exists), and their combination at inference | does combining separate policies fill it |
| 5 (optional) | rgbdt with pretrained image encoders, three seeds | is the encoder the obstacle |
| 6 | the list of baselines and the comparison rules for a method are frozen | |

Settings of every run: those of the `L0_vis` baselines (default config, 101 epochs, last checkpoint, 1320 test
episodes). Stages 3 to 5 are specified here before they start.

**Rules for all runs, fixed now.** (1) The same information for all: sensor images, joint values and the demo
actions; the arm's kinematics and the camera calibration are allowed to any run; nothing else of the simulator, except
in stage 1, whose policies are references and not baselines. (2) Default settings; nothing is selected on the test
sets. (3) Three seeds for every baseline that is compared; two policies are compared on the same episodes. (4) A
method is compared, cell by cell, with the best baseline of that cell (mean over seeds, over the baselines of stages
2 to 5). (5) "Better in nearly every case" means: no cell in which the method is below that best baseline with
p < 0.05 (pooled sign test on the same episodes), and a macro average above every baseline.

**Stage 1: references.** `L0_vis_{rgb,depth,thermal}_430demos_shown`: plain Diffusion Policy with that sensor's image
and the joints only (new variants `depth` and `thermal` of `policy/train.sh`), trained on the 430 demos of the cells
where the sensor shows the glass fully or faintly (rgb: cells 0, 2, 3, 6, 7, 10; depth: 0, 1, 3, 5, 8, 10; thermal:
0, 1, 2, 4, 9, 10; `policy/subset_demos.py`), seed 42. Read-out, fixed now: the reference of a cell is the highest
success among the policies whose sensor shows the glass in it. A cell with a reference of at least 0.30 counts as
shown to be solvable. If the reference of a single-sensor cell (4 to 9) is below 0.15, that sensor's level in that
cell is too weak for a policy; this is reported before anything else is concluded from the level, and whether the
level is changed is the user's decision.

**Stage 1, addition: single-cell references for thermal** (specified 11/21 04:50, before the runs; approved by the
user after the first stage-1 results). With the 430 demos where its sensor shows the glass fully or faintly, the
thermal-only reference reaches 0.20-0.22 where thermal shows the glass and 0.12-0.16 where faintly (depth-only:
0.60-0.69 and 0.45-0.48; rgb-only, first test set: 0.27-0.37 and 0.27-0.28). On `L0_grid` a policy trained on 500
demos of the thermal-only cell reached 0.575 there, so the mixed training set of the reference may be what holds it
down, not the cell. `L0_vis_thermal_500demos_cell4` and `L0_vis_thermal_500demos_cell9`: the thermal-only policy on
500 demos of that cell alone (`collect_demos.py --cells`), seed 42. Read-out, fixed now, on the cell the policy was
trained for (120 episodes): at least 0.30, the cell is solvable and the stage-1 recipe understates it; below 0.15,
the thermal level of that cell is too weak for a policy and the user decides whether the level changes; between,
partial. Single-cell references for the other four single-sensor cells are decided after this.

**Stage 2: seeds.** `L0_vis_{rgbdt,rgbd,rgbt,rgb}_700demos_seed{43,44}`. Reported: per cell and for the macro
average the mean over the three seeds and their range, and the pooled tables.

**Stage 3: random sensor dropout** (specified 11/21 02:00, before its runs).
`L0_vis_rgbdt_700demos_drop`, `..._drop_seed43`, `..._drop_seed44` (`TRAIN=policy/train_dropout.sh`): the plain rgbdt
policy; in training each of the three images of a sample (both observation steps) is blank with probability 0.3,
independently, and at least one stays. Validation and evaluation see all sensors. The probability is not tuned.
Read-out, fixed now: per cell and for the macro average against the plain rgbdt of the same seed on the same episodes.
In a cell without depth (2, 4, 6, 7, 9) the floor counts as filled if the mean over the three seeds is at least 0.30,
and as raised if the policy is above the plain rgbdt there with p < 0.05 (pooled sign test).

**Stage 4: separate policies combined at inference** (specified 11/21 02:05, before its runs).
`L0_vis_depth_700demos` and `L0_vis_thermal_700demos`: plain single-sensor policies trained on all 700 demos, seed 42
(the rgb one exists). They are baselines of their own, and the three are combined with `policy/eval_compose.py`: the
mean of their noise predictions at every denoising step, weights 1/3 each, nothing selected
(`L0_vis_compose_mean_700demos`, same test episodes). Read-out as in stage 3. Seeds 43 and 44 of the two new members
are added only if the combination, with seed 42, has a macro average within 0.05 of the best baseline or above it;
otherwise it is reported with one seed.

## Method probe: grasp maps per sensor from the demos alone (11/20; perception only, no policy yet)

Asked by the user: a method on the rgbdt policy that uses the sensors by situation, with nothing but sensor images
and joint values. This is a feasibility check of its first stage, not a benchmark result.

**Idea.** The demo itself says where the glass was: the tool position at the first step that closes the gripper (arm
kinematics), projected into the image with the camera calibration. Each sensor learns a distribution over the 96 x 96
pixels of where the grasp will be (a small fully convolutional network: local receptive field, no global pooling,
cross-entropy, image shifts). No cell, visibility flag or mask is used. A sensor's own reliability is learned from
maps on demos it was not trained on (4 folds): did the peak hit the grasp pixel, predicted from four numbers of the
map. Training: the 700 demos of `L0_vis`, first four frames; evaluation: 700 demos of other seeds, first frame
(`research/heatmap_probe.py`, [`method_probe/heatmap_maps.txt`](method_probe/heatmap_maps.txt)).

| sensor | pixel error where it shows the glass / faintly / not (median) | peak within 1.5 px | trained only where it shows the glass (uses the cells) | reliability: shows against blind (AUROC) |
|---|---|---|---|---|
| rgb | 0.20 / 0.25 / 3.91 | 0.99 / 0.97 / 0.11 | 0.17 / 0.16 | 0.924 |
| depth | 0.27 / 0.44 / 3.64 | 0.99 / 0.94 / 0.12 | 0.31 / 0.47 | 0.940 |
| thermal | 0.35 / 0.68 / 4.45 | 0.95 / 0.76 / 0.07 | 0.33 / 0.57 | 0.908 |

(1 px is about 1.9 x 1.3 cm on the table.)

- Trained on the mixed demos, every sensor finds the grasp pixel where it shows the glass as well as a network
  trained only on the demos where it does. The demos where a sensor is blind do not hurt it.
- Taking the sensor with the highest reliability gives a peak within 1.5 px in 0.90 of all held-out demos (per cell
  0.74-0.99, thermal-faint cell 0.58); depth alone 0.65. The product of the three maps is worse (0.68, with the
  reliabilities 0.74): a blind sensor still puts its peak somewhere and its reliability is not low enough (median
  0.39-0.49). The fusion rule is open.
- The same supervision as a regression of the grasp point from the whole image (ResNet trunk, soft-argmax keypoints,
  [`method_probe/global_regression.txt`](method_probe/global_regression.txt), `research/hindsight_probe.py`) works
  for depth only: where the sensor shows the glass 1.6 cm for depth, 4.1 cm for rgb and 3.0 cm for thermal, against
  1.6 / 2.0 / 2.1 cm when trained only on such demos; reweighting by out-of-fold error does not recover it. So the
  harm of mixed demos is not special to the diffusion loss, and a local architecture removes it.

Not done: a policy that acts on the fused map, and its success on the benchmark.

## Files

- `figure_cells.png`; `thermal_contrast_sweep.{txt,json}`; `real_thermal_contrast.{txt,json}`; `teacher_check.txt`;
  `cells_check_L0_bench_net.{txt,json}` (check 3), `cells_check_L0_grid_net.{txt,json}` (check 4).
- Baselines: `L0_grid_{rgbdt,rgbd,rgbt,rgb}_500demos/` (`train_logs.json.txt`, `curve.json`, `cells.json`, and per
  test set `eval_log.json`, `episodes.json`); `baselines_cells.txt`.
- Repair round: `policy/subset_demos.py` (the demos of chosen cells of a set), `sim/collect_demos.py --cells`
  (collect the seeds of chosen cells only), `ZARR` of `scripts/run_experiment.sh` (train on such a set); the
  defaults of both scripts are unchanged.
- Level `L0_grid_clear`: `sim/configs/L0_grid_clear.yaml`, `sim/check_clear.py`, `clear_check.txt`.
- "rgb shows" level: `sim/configs/L0_grid_clear_a0{4,5}.yaml`, `L0_grid_clear_a0{4,5}_rgb_330demos_rgbshown/`,
  `alpha_cells.txt`. Faint level: `policy/sweep_visibility.py`, `visibility_sweep.{txt,json}`.
- Baselines of the final level: `L0_vis_{rgbdt,rgbd,rgbt,rgb}_700demos/`, `vis_cells.txt`. Method probe:
  `research/{heatmap,hindsight}_probe.py`, `method_probe/`.
- Final level: `sim/configs/L0_vis.yaml`, `faint_calibration.txt`, `teacher_check_L0_vis.txt`; demo set
  `results/data/sim_demos_L0_vis_700.zarr` (not in git).
- Repair round and `L0_grid_clear` runs: `L0_grid_{rgb_330demos_rgbshown,rgb_1000demos,rgbt_500demos_cell4,rgbt_1000demos}/`,
  `L0_grid_clear_{rgbdt,rgbd,rgbt,rgb}_500demos/` (same files as the baselines); `repair_cells.txt`.
- Not in git: the demo sets `results/data/sim_demos_L0_grid_{500,1000,330_rgbshown,500_cell4}.zarr` and
  `sim_demos_L0_grid_clear_500.zarr`; the
  segmentation frames, network and logs under `data/grid_bench/`.
