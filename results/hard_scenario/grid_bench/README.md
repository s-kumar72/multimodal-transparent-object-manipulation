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

## Files

- `figure_cells.png`; `thermal_contrast_sweep.{txt,json}`; `real_thermal_contrast.{txt,json}`; `teacher_check.txt`;
  `cells_check_L0_bench_net.{txt,json}` (check 3), `cells_check_L0_grid_net.{txt,json}` (check 4).
- Baselines: `L0_grid_{rgbdt,rgbd,rgbt,rgb}_500demos/` (`train_logs.json.txt`, `curve.json`, `cells.json`, and per
  test set `eval_log.json`, `episodes.json`); `baselines_cells.txt`.
- Repair round: `policy/subset_demos.py` (the demos of chosen cells of a set), `sim/collect_demos.py --cells`
  (collect the seeds of chosen cells only), `ZARR` of `scripts/run_experiment.sh` (train on such a set); the
  defaults of both scripts are unchanged.
- Not in git: the demo sets `results/data/sim_demos_L0_grid_{500,1000,330_rgbshown,500_cell4}.zarr`; the
  segmentation frames, network and logs under `data/grid_bench/`.
