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

### Results of stage 1 (11/21 08:40; seed 42, 1320 test episodes per run, 120 per cell)

Success per cell; in brackets the cells where the policy's sensor does not show the glass. Reference: the highest
value among the policies whose sensor shows the glass in the cell, fully or faintly. Best plain baseline: the highest
of rgbdt, rgbd, rgbt and rgb (seed 42). Per run: `cells.json` in its folder.

| cell | rgb / depth / thermal | rgb only, 430 | depth only, 430 | thermal only, 430 | thermal only, 500 of cell 4 | thermal only, 500 of cell 9 | reference | best plain baseline |
|---|---|---|---|---|---|---|---|---|
| 0 | shows / shows / shows | 0.283 | 0.658 | 0.217 | 0.442 | 0.025 | **0.658** | 0.675 |
| 1 | blind / shows / shows | (0.050) | 0.658 | 0.200 | 0.375 | 0.025 | **0.658** | 0.633 |
| 2 | shows / blind / shows | 0.317 | (0.000) | 0.208 | 0.408 | 0.042 | **0.408** | 0.192 |
| 3 | shows / shows / blind | 0.258 | 0.692 | (0.017) | (0.017) | (0.033) | **0.692** | 0.633 |
| 4 | blind / blind / shows | (0.025) | (0.008) | 0.217 | 0.417 | 0.017 | **0.417** | 0.058 |
| 5 | blind / shows / blind | (0.058) | 0.600 | (0.008) | (0.008) | (0.050) | **0.600** | 0.550 |
| 6 | shows / blind / blind | 0.392 | (0.017) | (0.008) | (0.008) | (0.017) | **0.392** | 0.183 |
| 7 | faint / blind / blind | 0.267 | (0.017) | (0.025) | (0.008) | (0.017) | **0.267** | 0.108 |
| 8 | blind / faint / blind | (0.033) | 0.450 | (0.017) | (0.017) | (0.033) | **0.450** | 0.442 |
| 9 | blind / blind / faint | (0.025) | (0.017) | 0.158 | 0.192 | 0.000 | **0.192** | 0.067 |
| 10 | faint / faint / faint | 0.267 | 0.483 | 0.117 | 0.167 | 0.008 | **0.483** | 0.350 |
| | **mean of the eleven cells** | 0.180 | 0.327 | 0.108 | 0.187 | 0.024 | **0.474** | 0.354 |

Read-out by the rules fixed before the runs:

- **Nine of the eleven cells are shown to be solvable** (reference at least 0.30): cells 0 to 6, 8 and 10. Not shown
  to be solvable: cell 7 (rgb faint, 0.267) and cell 9 (thermal faint, 0.192). By the stage-1 rule no single-sensor
  cell has a reference below 0.15.
- **Cell 4 (thermal only) is solvable, and the first recipe understated it.** The thermal-only policy on 500 demos of
  cell 4 reaches 0.417 there (0.217 with the 430 demos where thermal shows the glass fully or faintly), and 0.375 to
  0.442 in every cell where thermal shows the glass fully.
- **Cell 9 (thermal faint): 0.000 (0 of 120) for the policy trained on 500 demos of that cell, below 0.15.** By the
  rule of the addition the thermal level of that cell is too weak for a policy, and whether the level changes is the
  user's decision. The level is not changed here. That policy learned nothing: at most 0.05 in every cell, also where
  thermal shows the glass fully.

Observations outside the read-out:

- The same faint level is read, in part, by a policy that learned where thermal is clear: 0.192 in cell 9 for the
  policy trained on cell 4 alone, 0.158 for the one trained on the 430 demos. A policy that is given only faint demos
  learns nothing from them. The thermal level of cell 9 is therefore weak for learning, not without information for a
  trained policy (the grasp maps of the method probe hit in 0.76 of the faint thermal frames).
- The 430 demos of the first thermal reference are 330 with thermal fully visible and 100 with thermal faint (cells 9
  and 10); the thermal image does not depend on the other two sensors' flags. With them the policy reaches about half
  of the cell-4 policy where thermal shows the glass fully (0.200-0.217 against 0.375-0.442). The two training sets
  also differ in size (430 and 500), so this is an indication, not a controlled comparison. The rgb and depth
  references were trained with the same recipe (100 faint demos of 430) and may be understated in the same way.
- **Where depth shows the glass the plain baselines are at the reference; without depth they are far below it.**
  Cells with depth (0, 1, 3, 5, 8, 10): reference 0.590 on average, best plain baseline 0.547. Cells without depth
  (2, 4, 6, 7, 9): reference 0.335, best plain baseline 0.122. The room for a method is in those five cells.

### Thermal faint level kept; references on the fully visible demos (11/21 09:00, written before the runs)

Decision of the user (11/21): the thermal faint level stays as it is (2.7-3.3 C from the table). Cell 9 is kept and
reported with its reference; nothing of the level, the training set or the test sets changes.

Approved by the user: `L0_vis_{rgb,depth,thermal}_330demos_full`, the single-sensor policy of stage 1 trained on the
330 of the 700 training demos where its sensor shows the glass fully (rgb: cells 0, 2, 3, 6; depth: 0, 1, 3, 5;
thermal: 0, 1, 2, 4; `policy/subset_demos.py`), seed 42, settings as before. Question: do the 100 faint demos of the
430-demo recipe hold a reference down? For thermal the indication above is not controlled; for rgb and depth it has
not been looked at.

Read-out, fixed now. (1) Per sensor, the 330-demo policy against the 430-demo policy on the same test episodes,
pooled over the four cells where the sensor shows the glass fully (480 episodes) and, separately, over the two where
it shows it faintly (240 episodes); sign test. Above with p < 0.05 where the sensor shows the glass fully: the faint
demos lowered that reference (the 330-demo policy has fewer demos, so the number of demos cannot be the reason). No
difference: the recipe was not what held it down. (2) The reference of a cell remains the highest value among all
reference policies whose sensor shows the glass in it, these three included; the thresholds 0.30 and 0.15 stay. One
seed.

### Results of stage 4 (11/21 17:00; seed 42, 1320 test episodes per run, 120 per cell)

The three single-sensor policies trained on all 700 demos, and their equal-weight combination at inference
(`L0_vis_compose_mean_700demos`). Per run: `cells.json` in its folder.

| cell | rgb / depth / thermal | rgb only | depth only | thermal only | **combination** | plain rgbdt | plain rgbd |
|---|---|---|---|---|---|---|---|
| 0 | shows / shows / shows | 0.100 | 0.450 | 0.025 | **0.217** | 0.675 | 0.633 |
| 1 | blind / shows / shows | 0.025 | 0.433 | 0.033 | **0.242** | 0.567 | 0.633 |
| 2 | shows / blind / shows | 0.092 | 0.033 | 0.042 | **0.050** | 0.092 | 0.092 |
| 3 | shows / shows / blind | 0.100 | 0.400 | 0.017 | **0.225** | 0.633 | 0.600 |
| 4 | blind / blind / shows | 0.017 | 0.067 | 0.042 | **0.017** | 0.050 | 0.017 |
| 5 | blind / shows / blind | 0.025 | 0.400 | 0.008 | **0.208** | 0.492 | 0.550 |
| 6 | shows / blind / blind | 0.075 | 0.050 | 0.008 | **0.017** | 0.100 | 0.108 |
| 7 | faint / blind / blind | 0.050 | 0.058 | 0.017 | **0.075** | 0.058 | 0.050 |
| 8 | blind / faint / blind | 0.042 | 0.267 | 0.000 | **0.150** | 0.375 | 0.442 |
| 9 | blind / blind / faint | 0.042 | 0.017 | 0.025 | **0.033** | 0.067 | 0.058 |
| 10 | faint / faint / faint | 0.067 | 0.325 | 0.000 | **0.142** | 0.350 | 0.292 |
| | **macro average, eleven cells** | **0.058** | **0.227** | **0.020** | **0.125** | **0.314** | **0.316** |
| | cells without depth (2, 4, 6, 7, 9) | 0.055 | 0.045 | 0.027 | 0.038 | 0.073 | 0.065 |
| | cells with depth (0, 1, 3, 5, 8, 10) | 0.060 | 0.379 | 0.014 | 0.197 | 0.515 | 0.525 |

Read-out by the rules fixed before the runs:

- **Combining the separate policies does not fill the floor.** In the five cells without depth the combination has
  0.017 to 0.075 (mean 0.038); none reaches 0.30 and none is above the plain rgbdt with p < 0.05 (the only positive
  difference is +0.017 in cell 7, p = 0.75).
- **It is below the plain baselines everywhere else.** Macro average 0.125 against 0.314 for the plain rgbdt on the
  same episodes (only the combination succeeds in 52 episodes, only rgbdt in 302; p = 5.6e-44), and below its own
  depth member (0.227; p = 2.7e-17).
- The macro average is not within 0.05 of the best baseline (0.316), so the combination is reported with one seed
  and seeds 43 and 44 of its members are not run.
- **The members as baselines of their own.** Depth only: 0.227, with 0.40 to 0.45 where depth shows the glass fully.
  Thermal only: 0.020, at most 0.042 in any cell; it learned nothing. Rgb only (from the baselines): 0.058.

Observations outside the read-out:

- Two of the three members did not learn their sensor, so the mean of the noise predictions is the depth policy
  diluted by two policies that do not see the glass: about half of the depth member where depth shows the glass
  (0.21-0.24 against 0.40-0.45). Combining at inference cannot supply what the members did not learn.
- **A single-sensor policy is lowered by the demos in which its sensor does not show the glass, for depth as well.**
  Where the sensor shows the glass fully: depth only 0.60-0.69 on the 430 demos where depth shows the glass, 0.40-0.45
  on all 700 (270 more demos, all with depth blind); thermal only 0.20-0.22 on its 430, 0.03-0.04 on all 700; rgb only
  0.26-0.39 on its 430, 0.08-0.10 on all 700. More demos, lower success, for all three sensors.

### Results of stage 2 (11/21 17:25; seeds 42, 43 and 44, 1320 test episodes per run)

Mean over the three seeds and, in brackets, their range. Best baseline: the highest mean of the cell; this is what a
method is compared with (rule 4). Pooled tables and the paired comparisons: [`seeds_cells.txt`](seeds_cells.txt).

| cell | rgb / depth / thermal | rgbdt | rgbd | rgbt | rgb | best baseline |
|---|---|---|---|---|---|---|
| 0 | shows / shows / shows | 0.611 (0.567-0.675) | 0.583 (0.558-0.633) | 0.100 (0.067-0.167) | 0.103 (0.083-0.125) | 0.611 (rgbdt) |
| 1 | blind / shows / shows | 0.508 (0.450-0.567) | 0.639 (0.633-0.650) | 0.025 (0.025-0.025) | 0.036 (0.025-0.058) | 0.639 (rgbd) |
| 2 | shows / blind / shows | 0.142 (0.092-0.175) | 0.094 (0.083-0.108) | 0.175 (0.133-0.200) | 0.100 (0.092-0.117) | 0.175 (rgbt) |
| 3 | shows / shows / blind | 0.625 (0.592-0.650) | 0.625 (0.583-0.692) | 0.158 (0.108-0.225) | 0.108 (0.083-0.142) | 0.625 (rgbdt) |
| 4 | blind / blind / shows | 0.050 (0.050-0.050) | 0.025 (0.017-0.033) | 0.039 (0.017-0.058) | 0.025 (0.008-0.050) | 0.050 (rgbdt) |
| 5 | blind / shows / blind | 0.442 (0.367-0.492) | 0.475 (0.433-0.550) | 0.056 (0.025-0.075) | 0.039 (0.025-0.050) | 0.475 (rgbd) |
| 6 | shows / blind / blind | 0.092 (0.083-0.100) | 0.103 (0.083-0.117) | 0.133 (0.092-0.183) | 0.103 (0.075-0.117) | 0.133 (rgbt) |
| 7 | faint / blind / blind | 0.064 (0.058-0.075) | 0.072 (0.050-0.092) | 0.092 (0.067-0.108) | 0.078 (0.050-0.108) | 0.092 (rgbt) |
| 8 | blind / faint / blind | 0.319 (0.267-0.375) | 0.381 (0.317-0.442) | 0.033 (0.025-0.042) | 0.036 (0.033-0.042) | 0.381 (rgbd) |
| 9 | blind / blind / faint | 0.061 (0.042-0.075) | 0.064 (0.050-0.083) | 0.050 (0.042-0.058) | 0.039 (0.033-0.042) | 0.064 (rgbd) |
| 10 | faint / faint / faint | 0.308 (0.275-0.350) | 0.333 (0.292-0.367) | 0.108 (0.092-0.117) | 0.081 (0.067-0.108) | 0.333 (rgbd) |
| | **macro average, eleven cells** | **0.293** (0.277-0.314) | **0.309** (0.298-0.316) | **0.088** (0.070-0.112) | **0.068** (0.058-0.074) | 0.325 |
| | cells without depth (2, 4, 6, 7, 9) | 0.082 | 0.072 | 0.098 | 0.069 | 0.103 |
| | cells with depth (0, 1, 3, 5, 8, 10) | 0.469 | 0.506 | 0.080 | 0.067 | 0.511 |
| | cells 0-6 | 0.353 | 0.363 | 0.098 | 0.073 | 0.387 |
| | cells 7-10 | 0.188 | 0.212 | 0.071 | 0.058 | 0.217 |
| | macro average, seeds 42 / 43 / 44 | 0.314 / 0.288 / 0.277 | 0.316 / 0.311 / 0.298 | 0.112 / 0.083 / 0.070 | 0.058 / 0.072 / 0.074 | |

- **The baseline numbers are stable.** The macro average moves by at most 0.04 between seeds (standard deviation
  0.02 for rgbdt and rgbt, 0.01 for rgbd and rgb), and the order of the four baselines is the same for every seed
  except rgbt and rgb with seed 44 (0.070 and 0.074).
- **With three seeds rgbdt is not above rgbd; it is slightly below.** 0.293 against 0.309, pooled on the same
  episodes: only rgbdt succeeds in 369, only rgbd in 431, p = 0.031. By cell: below where depth and thermal show the
  glass and rgb does not (cell 1, -0.131, p = 1.7e-05) and where depth is faint (cell 8, -0.061, p = 0.035), above in
  cell 2 (rgb and thermal, +0.047, p = 0.027), which stays near the floor (0.142). The thermal image does not help the
  plain policy.
- **The floor holds with three seeds.** In the five cells without depth the best baseline of each cell has 0.050 to
  0.175 (mean 0.103), against a reference of 0.192 to 0.417 (mean 0.335). Where depth shows the glass the best
  baseline has 0.333 to 0.639 (mean 0.511; reference 0.590).
- rgbt is above rgb by 0.020 (p = 0.00015), mostly from cells 2 and 3 (+0.075, p = 0.0013, and +0.050, p = 0.033);
  both stay below 0.18 in every cell.
- Best baseline of every cell: mean 0.325 (best single baseline: rgbd, 0.309).

### Results of stage 3 (11/21 20:00; seeds 42, 43 and 44, 1320 test episodes per run)

rgbdt with random sensor dropout in training (p = 0.3 per image), against the plain rgbdt of the same seed on the same
episodes. Mean over the three seeds and, in brackets, their range; differences and sign tests pooled over the three
seeds (360 episodes per cell). Per run: `cells.json` in its folder.

| cell | rgb / depth / thermal | rgbdt with dropout | plain rgbdt | difference | only dropout succeeds | only plain succeeds | sign test p |
|---|---|---|---|---|---|---|---|
| 0 | shows / shows / shows | 0.433 (0.417-0.458) | 0.611 | -0.178 | 29 | 93 | 5.2e-09 |
| 1 | blind / shows / shows | 0.433 (0.392-0.492) | 0.508 | -0.075 | 49 | 76 | 0.020 |
| 2 | shows / blind / shows | 0.081 (0.058-0.100) | 0.142 | -0.061 | 18 | 40 | 0.0054 |
| 3 | shows / shows / blind | 0.408 (0.367-0.433) | 0.625 | -0.217 | 30 | 108 | 1.6e-11 |
| 4 | blind / blind / shows | 0.033 (0.025-0.042) | 0.050 | -0.017 | 10 | 16 | 0.33 |
| 5 | blind / shows / blind | 0.467 (0.442-0.508) | 0.442 | +0.025 | 57 | 48 | 0.44 |
| 6 | shows / blind / blind | 0.106 (0.075-0.125) | 0.092 | +0.014 | 27 | 22 | 0.57 |
| 7 | faint / blind / blind | 0.053 (0.050-0.058) | 0.064 | -0.011 | 16 | 20 | 0.62 |
| 8 | blind / faint / blind | 0.333 (0.275-0.367) | 0.319 | +0.014 | 51 | 46 | 0.68 |
| 9 | blind / blind / faint | 0.036 (0.033-0.042) | 0.061 | -0.025 | 12 | 21 | 0.16 |
| 10 | faint / faint / faint | 0.328 (0.267-0.367) | 0.308 | +0.019 | 51 | 44 | 0.54 |
| | **macro average, eleven cells** | **0.246** (0.225-0.267) | **0.293** | -0.046 | 350 | 534 | 6.5e-10 |
| | cells without depth (2, 4, 6, 7, 9) | 0.062 | 0.082 | | | | |
| | cells with depth (0, 1, 3, 5, 8, 10) | 0.400 | 0.469 | | | | |
| | macro average, seeds 42 / 43 / 44 | 0.247 / 0.267 / 0.225 | 0.314 / 0.288 / 0.277 | | | | |

Read-out by the rules fixed before the runs:

- **Sensor dropout does not fill the floor and does not raise it.** In the five cells without depth the mean over
  the three seeds is 0.033 to 0.106 (mean 0.062); none reaches 0.30, and in none is the policy above the plain rgbdt
  with p < 0.05 (the largest positive difference is +0.014 in cell 6, p = 0.57). In cell 2 it is below the plain
  rgbdt (-0.061, p = 0.0054).
- **It is below the plain rgbdt overall.** Macro average 0.246 against 0.293 (p = 6.5e-10), with every seed below its
  plain run (-0.067, -0.020, -0.052; p = 2.7e-07, 0.13, 8.2e-05). The loss is where two or three sensors show the
  glass fully: cells 0, 1 and 3 (-0.178, -0.075, -0.217) and cell 2. Where one sensor shows the glass, or all are
  faint (cells 4 to 10), there is no difference with p < 0.05 in either direction.
- As a baseline of its own it is the best of no cell: its three-seed mean is below the best plain baseline in every
  cell (closest: cell 5, 0.467 against 0.475, and cell 10, 0.328 against 0.333). Against rgbd, pooled: 0.246 and
  0.309, p = 7.3e-17.

### Results of the references on the fully visible demos (11/21 20:00; seed 42, 1320 test episodes per run)

`L0_vis_{rgb,depth,thermal}_330demos_full`: success per cell, in brackets the cells where the policy's sensor does not
show the glass. Per run: `cells.json` in its folder.

| cell | rgb / depth / thermal | rgb only, 330 | depth only, 330 | thermal only, 330 | reference before | **reference now** | best baseline (three seeds) |
|---|---|---|---|---|---|---|---|
| 0 | shows / shows / shows | 0.367 | 0.742 | 0.425 | 0.658 | **0.742** | 0.611 |
| 1 | blind / shows / shows | (0.033) | 0.817 | 0.333 | 0.658 | **0.817** | 0.639 |
| 2 | shows / blind / shows | 0.400 | (0.067) | 0.375 | 0.408 | **0.408** | 0.175 |
| 3 | shows / shows / blind | 0.333 | 0.717 | (0.025) | 0.692 | **0.717** | 0.625 |
| 4 | blind / blind / shows | (0.033) | (0.008) | 0.383 | 0.417 | **0.417** | 0.050 |
| 5 | blind / shows / blind | (0.042) | 0.717 | (0.033) | 0.600 | **0.717** | 0.475 |
| 6 | shows / blind / blind | 0.317 | (0.025) | (0.017) | 0.392 | **0.392** | 0.133 |
| 7 | faint / blind / blind | 0.142 | (0.025) | (0.042) | 0.267 | **0.267** | 0.092 |
| 8 | blind / faint / blind | (0.033) | 0.292 | (0.058) | 0.450 | **0.450** | 0.381 |
| 9 | blind / blind / faint | (0.042) | (0.033) | 0.175 | 0.192 | **0.192** | 0.064 |
| 10 | faint / faint / faint | 0.133 | 0.300 | 0.158 | 0.483 | **0.483** | 0.333 |
| | **mean of the eleven cells** | 0.170 | 0.340 | 0.184 | 0.474 | **0.509** | 0.325 |
| | cells without depth (2, 4, 6, 7, 9) | | | | 0.335 | 0.335 | 0.103 |
| | cells with depth (0, 1, 3, 5, 8, 10) | | | | 0.590 | 0.654 | 0.511 |

The 330-demo policy against the 430-demo policy of the same sensor, on the same test episodes:

| sensor | cells | episodes | 330 demos | 430 demos | difference | only 330 succeeds | only 430 succeeds | sign test p |
|---|---|---|---|---|---|---|---|---|
| rgb | shows fully: 0, 2, 3, 6 | 480 | 0.354 | 0.312 | +0.042 | 77 | 57 | 0.10 |
| rgb | shows faintly: 7, 10 | 240 | 0.138 | 0.267 | -0.129 | 16 | 47 | 0.00012 |
| depth | shows fully: 0, 1, 3, 5 | 480 | 0.748 | 0.652 | +0.096 | 69 | 23 | 1.7e-06 |
| depth | shows faintly: 8, 10 | 240 | 0.296 | 0.467 | -0.171 | 27 | 68 | 3.1e-05 |
| thermal | shows fully: 0, 1, 2, 4 | 480 | 0.379 | 0.210 | +0.169 | 118 | 37 | 4.7e-11 |
| thermal | shows faintly: 9, 10 | 240 | 0.167 | 0.138 | +0.029 | 26 | 19 | 0.37 |

Read-out by the rules fixed before the runs:

- **Depth and thermal: the faint demos lowered the reference.** Where the sensor shows the glass fully the policy
  with 100 demos fewer is above the 430-demo policy: depth 0.748 against 0.652 (p = 1.7e-06), thermal 0.379 against
  0.210 (p = 4.7e-11). The number of demos cannot be the reason.
- **Rgb: no difference** where rgb shows the glass fully (0.354 against 0.312, p = 0.10). The recipe was not what
  held the rgb reference down.
- **References.** They rise in the four cells where depth shows the glass fully (cells 0, 1, 3 and 5: 0.742, 0.817,
  0.717, 0.717) and are unchanged in the other seven; mean 0.509 (before 0.474). The same nine cells are shown to be
  solvable. Cell 7 (rgb faint, 0.267) and cell 9 (thermal faint, 0.192) stay below 0.30, and no single-sensor cell
  is below 0.15.

Observations outside the read-out:

- **The faint demos are needed for the faint cells of rgb and depth, and cost success in the clear ones.** Without
  them the policy loses where its sensor is faint (rgb -0.129, depth -0.171) and, for depth and thermal, gains where
  it is clear. For thermal the faint demos gave nothing in the faint cells either (0.138 with them, 0.167 without).
  One policy per sensor trained on everything is therefore not the best at both; the reference takes the better of
  the two per cell.
- The thermal-only policy on the 330 demos is level with the one trained on 500 demos of cell 4 alone (0.379 and
  0.410 where thermal shows the glass fully, p = 0.29): the earlier gap between the 430-demo recipe and the cell-4
  policy came from the faint demos, not from mixing cells.
- **The plain baselines are now below the reference where depth shows the glass as well.** Cells 0, 1, 3 and 5: the
  depth-only policy on 330 demos has 0.748, the plain rgbd of seed 42 0.604 and the plain rgbdt 0.592 on the same
  episodes (p = 5.1e-09 and 4.3e-10). A policy with one image and 330 demos is above the policies with two or three
  images and all 700 demos. In the cells without depth, against the plain rgbdt and rgbt of seed 42: thermal only
  0.379 in cells 2 and 4 against 0.071 and 0.125; rgb only 0.358 in cells 2 and 6 against 0.096 and 0.188.
- The reference is the highest of up to eight policies with 120 episodes per cell and one seed, so it leans high; the
  best baseline is a mean over three seeds. The gap between them (0.07 to 0.37 per cell, 0.18 on average) is read
  with that in mind.

### State after stages 1 to 4 (11/21 20:00)

| | macro average | cells without depth | cells with depth | seeds |
|---|---|---|---|---|
| reference (best single-sensor policy of the cell; uses the cells) | 0.509 | 0.335 | 0.654 | 1 |
| best baseline of each cell | 0.325 | 0.103 | 0.511 | 3 |
| rgbd | 0.309 | 0.072 | 0.506 | 3 |
| rgbdt | 0.293 | 0.082 | 0.469 | 3 |
| rgbdt with sensor dropout | 0.246 | 0.062 | 0.400 | 3 |
| depth only, all 700 demos | 0.227 | 0.045 | 0.379 | 1 |
| combination of the three single-sensor policies | 0.125 | 0.038 | 0.197 | 1 |
| rgbt | 0.088 | 0.098 | 0.080 | 3 |
| rgb | 0.068 | 0.069 | 0.067 | 3 |
| thermal only, all 700 demos | 0.020 | 0.027 | 0.014 | 1 |

No baseline and neither of the two known remedies (sensor dropout, combining separate policies) reaches the
reference; the best baseline of every cell is one of the four plain policies. Open before stage 6: whether stage 5
(pretrained image encoders) is run, which is the user's decision. The baselines with one seed are below a three-seed
baseline in every cell except cell 4 (depth only 0.067, rgbdt 0.050; both at the floor).

### Stage 5: pretrained image encoders (specified 11/22 02:37, before its runs; approved by the user)

`L0_vis_rgbdt_700demos_imagenet`, `..._imagenet_seed43`, `..._imagenet_seed44`: the plain rgbdt policy with the
override `policy.obs_encoder.rgb_model.weights=IMAGENET1K_V1`, the "ImageNet" recipe of the diagnosis round
([`../diagnosis/README.md`](../diagnosis/README.md)): each of the three image encoders (rgb, depth, thermal) starts
from the ImageNet weights of ResNet-18 instead of random ones and is trained end to end as before. Nothing else
differs from the plain rgbdt: the architecture is unchanged (GroupNorm in place of BatchNorm, as in the default, so
the statistics of the pretrained normalization layers are not used), same optimizer and learning rate, 101 epochs,
last checkpoint, 1320 test episodes. No other pretrained recipe (frozen encoders, other weights, another learning
rate for the encoders) is tried and nothing is tuned.

Read-out, fixed now, as in stage 3: per cell and for the macro average against the plain rgbdt of the same seed on
the same episodes (pooled sign test over the three seeds). (1) In a cell without depth (2, 4, 6, 7, 9) the floor
counts as filled if the mean over the three seeds is at least 0.30, and as raised if the policy is above the plain
rgbdt there with p < 0.05. (2) The question of the stage: the random initialization of the encoders counts as an
obstacle if the macro average is above the plain rgbdt with p < 0.05; otherwise it is not what holds the plain
policy down. (3) With its three seeds the policy joins the baselines, and the best baseline of every cell (rule 4) is
taken again with it.

### Results of stage 5 (11/22 07:00; seeds 42, 43 and 44, 1320 test episodes per run)

rgbdt with the image encoders started from ImageNet weights, against the plain rgbdt of the same seed on the same
episodes. Mean over the three seeds and, in brackets, their range; differences and sign tests pooled over the three
seeds (360 episodes per cell). Per run: `cells.json` in its folder.

| cell | rgb / depth / thermal | rgbdt, ImageNet | plain rgbdt | difference | only ImageNet succeeds | only plain succeeds | sign test p | best baseline before | reference |
|---|---|---|---|---|---|---|---|---|---|
| 0 | shows / shows / shows | 0.781 (0.775-0.792) | 0.611 | +0.169 | 91 | 30 | 2.5e-08 | 0.611 | 0.742 |
| 1 | blind / shows / shows | 0.758 (0.750-0.775) | 0.508 | +0.250 | 112 | 22 | 1.0e-15 | 0.639 | 0.817 |
| 2 | shows / blind / shows | 0.286 (0.225-0.333) | 0.142 | +0.144 | 79 | 27 | 4.3e-07 | 0.175 | 0.408 |
| 3 | shows / shows / blind | 0.692 (0.650-0.733) | 0.625 | +0.067 | 81 | 57 | 0.050 | 0.625 | 0.717 |
| 4 | blind / blind / shows | 0.183 (0.175-0.192) | 0.050 | +0.133 | 59 | 11 | 4.5e-09 | 0.050 | 0.417 |
| 5 | blind / shows / blind | 0.658 (0.625-0.700) | 0.442 | +0.217 | 116 | 38 | 2.2e-10 | 0.475 | 0.717 |
| 6 | shows / blind / blind | 0.144 (0.108-0.167) | 0.092 | +0.053 | 42 | 23 | 0.025 | 0.133 | 0.392 |
| 7 | faint / blind / blind | 0.131 (0.092-0.150) | 0.064 | +0.067 | 45 | 21 | 0.0043 | 0.092 | 0.267 |
| 8 | blind / faint / blind | 0.486 (0.458-0.525) | 0.319 | +0.167 | 93 | 33 | 8.5e-08 | 0.381 | 0.450 |
| 9 | blind / blind / faint | 0.089 (0.050-0.117) | 0.061 | +0.028 | 28 | 18 | 0.18 | 0.064 | 0.192 |
| 10 | faint / faint / faint | 0.503 (0.467-0.558) | 0.308 | +0.194 | 104 | 34 | 1.9e-09 | 0.333 | 0.483 |
| | **macro average, eleven cells** | **0.428** (0.414-0.440) | **0.293** | +0.135 | 850 | 314 | 1.8e-57 | 0.325 | 0.509 |
| | cells without depth (2, 4, 6, 7, 9) | 0.167 | 0.082 | | | | | 0.103 | 0.335 |
| | cells with depth (0, 1, 3, 5, 8, 10) | 0.646 | 0.469 | | | | | 0.511 | 0.654 |
| | macro average, seeds 42 / 43 / 44 | 0.414 / 0.440 / 0.431 | 0.314 / 0.288 / 0.277 | | | | | | |

Read-out by the rules fixed before the runs:

- **(1) The floor is raised in four of the five cells without depth and filled in none.** Above the plain rgbdt with
  p < 0.05 in cells 2, 4, 6 and 7 (+0.144, +0.133, +0.053, +0.067); no difference in cell 9 (thermal faint, +0.028,
  p = 0.18). The three-seed means there are 0.089 to 0.286 (mean 0.167); none reaches 0.30.
- **(2) The random initialization of the encoders is an obstacle.** Macro average 0.428 against 0.293 for the plain
  rgbdt (p = 1.8e-57), with every seed above its plain run (+0.099, +0.152, +0.155), and above in every cell (p <
  0.05 in ten of the eleven; cell 3 with p = 0.0498, cell 9 not).
- **(3) It is now the best baseline of every cell.** Mean of the best baseline 0.428 (before 0.325). Against rgbd,
  the best single baseline so far, pooled: 0.428 and 0.309 (p = 4.3e-45).

Observations outside the read-out:

- **Where depth shows the glass this baseline is at the reference** (0.646 against 0.654 on average; above it in
  cells 0, 8 and 10, below in cells 1, 3 and 5; seed 42 against the depth-only policy on the 330 demos in cells 0, 1,
  3 and 5 on the same episodes: 0.706 and 0.748, p = 0.14). **Without depth it stays at half of the reference**
  (0.167 against 0.335), and below the single-sensor policy of the sensor that shows the glass in each of those cells
  (seed 42, same episodes: 0.208 against 0.379 for thermal only in cells 2 and 4, p = 2e-05; 0.192 against 0.358 for
  rgb only in cells 2 and 6, p = 6.4e-06; 0.092 against 0.267 in cell 7; 0.050 against 0.192 in cell 9).
- **The comparison is no longer like for like.** The other baselines (rgbd, rgbt, rgb, sensor dropout, the
  single-sensor policies) and all reference policies start from random encoders. Whether rgbd, rgbt or rgb with
  ImageNet encoders are above the ImageNet rgbdt in some cell, and how high the references are with ImageNet
  encoders, is not known. In the diagnosis round ImageNet encoders only sped up the start on other levels with 200
  demos; here they change the end result.
- In the records of the last gripper close (section below): where depth shows the glass fully the ImageNet rgbdt
  succeeds in 0.96 of the episodes in which the gripper is placed within 1.5 cm (plain rgbdt 0.75), with the same
  share of placed episodes (0.68 and 0.64). Without depth it places the gripper within 1.5 cm in 0.10 of the episodes
  (plain rgbdt 0.07; the single-sensor references 0.40 to 0.43): it grasps better, and does not find the glass more
  often.

### State after stages 1 to 5 (11/22 07:00)

| | macro average | cells without depth | cells with depth | seeds | encoders |
|---|---|---|---|---|---|
| reference (best single-sensor policy of the cell; uses the cells) | 0.509 | 0.335 | 0.654 | 1 | random |
| **rgbdt, ImageNet encoders (best baseline of every cell)** | **0.428** | **0.167** | **0.646** | 3 | ImageNet |
| rgbd | 0.309 | 0.072 | 0.506 | 3 | random |
| rgbdt | 0.293 | 0.082 | 0.469 | 3 | random |
| rgbdt with sensor dropout | 0.246 | 0.062 | 0.400 | 3 | random |
| depth only, all 700 demos | 0.227 | 0.045 | 0.379 | 1 | random |
| combination of the three single-sensor policies | 0.125 | 0.038 | 0.197 | 1 | random |
| rgbt | 0.088 | 0.098 | 0.080 | 3 | random |
| rgb | 0.068 | 0.069 | 0.067 | 3 | random |
| thermal only, all 700 demos | 0.020 | 0.027 | 0.014 | 1 | random |

Open before stage 6, for the user to decide: whether the list is frozen as it is, or the other baselines and the
references are first run with ImageNet encoders as well (see the second observation above).

### Stage 5, addition: the other baselines and the references with ImageNet encoders (specified 11/22 07:48, before the runs; decision of the user)

Decision of the user (11/22): the list is not frozen with one baseline on ImageNet encoders and everything else on
random ones; the comparison is made like for like first. Fifteen runs, all with the override of stage 5
(`policy.obs_encoder.rgb_model.weights=IMAGENET1K_V1`) and otherwise the settings of their counterparts with random
encoders (same demo sets, 101 epochs, last checkpoint, 1320 test episodes):

- **Baselines, three seeds:** `L0_vis_{rgbd,rgbt,rgb}_700demos_imagenet`, `..._imagenet_seed43`, `..._imagenet_seed44`.
- **References, seed 42:** `L0_vis_{rgb,depth,thermal}_330demos_full_imagenet` (the 330 demos where the sensor shows
  the glass fully) and `L0_vis_{rgb,depth,thermal}_430demos_shown_imagenet` (the 430 where it shows it fully or
  faintly), the two recipes of stage 1 that use only the 700 training demos.

Not rerun with ImageNet encoders: sensor dropout, the single-sensor policies on all 700 demos and their combination
(stages 3 and 4, both below the plain rgbdt with random encoders), and the thermal references on 500 demos of one
cell (demos outside the training set). They stay in the tables as they are.

Read-out, fixed now.
(1) Baselines: per cell and for the macro average the mean over the three seeds and their range, as in stage 2, and
each against its counterpart with random encoders on the same episodes. The ImageNet rgbdt against the ImageNet rgbd,
rgbt and rgb on the same episodes, pooled over the seeds (sign test): if rgbdt is above rgbd with p < 0.05 on the
macro average, the thermal image helps the plain policy once the encoders are pretrained; if not, the stage-2 finding
(rgbdt is not above rgbd) holds for ImageNet encoders too. The best baseline of every cell (rule 4) is taken again
over all baselines with three seeds.
(2) References: per sensor and recipe, the ImageNet policy against its counterpart with random encoders on the same
episodes, pooled over the four cells where the sensor shows the glass fully (480 episodes) and over the two where it
shows it faintly (240); sign test. The reference of a cell remains the highest value among all reference policies
whose sensor shows the glass in it, with random or ImageNet encoders; the thresholds 0.30 and 0.15 stay. One seed.
(3) Reported after both: per cell the best baseline against the reference, and where the best baseline is below the
single-sensor policy of a sensor that shows the glass (same seed, same episodes, sign test).

### Results of the references with ImageNet encoders (11/22 11:30; seed 42, 1320 test episodes per run)

First part of the addition to stage 5; the nine baseline runs are still running. Success per cell, in brackets the
cells where the policy's sensor does not show the glass. Per run: `cells.json` in its folder.

| cell | rgb / depth / thermal | rgb, 330 | rgb, 430 | depth, 330 | depth, 430 | thermal, 330 | thermal, 430 | reference before | **reference now** | rgbdt, ImageNet (three seeds) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | shows / shows / shows | 0.517 | 0.542 | 0.867 | 0.900 | 0.575 | 0.608 | 0.742 | **0.900** | 0.781 |
| 1 | blind / shows / shows | (0.050) | (0.058) | 0.858 | 0.875 | 0.567 | 0.475 | 0.817 | **0.875** | 0.758 |
| 2 | shows / blind / shows | 0.575 | 0.642 | (0.000) | (0.025) | 0.558 | 0.567 | 0.408 | **0.642** | 0.286 |
| 3 | shows / shows / blind | 0.567 | 0.508 | 0.867 | 0.817 | (0.025) | (0.008) | 0.717 | **0.867** | 0.692 |
| 4 | blind / blind / shows | (0.008) | (0.008) | (0.000) | (0.008) | 0.608 | 0.583 | 0.417 | **0.608** | 0.183 |
| 5 | blind / shows / blind | (0.042) | (0.050) | 0.858 | 0.800 | (0.000) | (0.008) | 0.717 | **0.858** | 0.658 |
| 6 | shows / blind / blind | 0.425 | 0.492 | (0.000) | (0.058) | (0.008) | (0.000) | 0.392 | **0.492** | 0.144 |
| 7 | faint / blind / blind | 0.242 | 0.333 | (0.000) | (0.050) | (0.042) | (0.025) | 0.267 | **0.333** | 0.131 |
| 8 | blind / faint / blind | (0.008) | (0.017) | 0.467 | 0.742 | (0.033) | (0.025) | 0.450 | **0.742** | 0.486 |
| 9 | blind / blind / faint | (0.008) | (0.025) | (0.017) | (0.008) | 0.217 | 0.383 | 0.192 | **0.383** | 0.089 |
| 10 | faint / faint / faint | 0.200 | 0.367 | 0.492 | 0.717 | 0.308 | 0.333 | 0.483 | **0.717** | 0.503 |
| | **mean of the eleven cells** | 0.240 | 0.277 | 0.402 | 0.455 | 0.267 | 0.274 | 0.509 | **0.674** | 0.428 |
| | cells without depth (2, 4, 6, 7, 9) | | | | | | | 0.335 | 0.492 | 0.167 |
| | cells with depth (0, 1, 3, 5, 8, 10) | | | | | | | 0.654 | 0.826 | 0.646 |

The ImageNet policy against its counterpart with random encoders, on the same test episodes:

| sensor | demos | cells | episodes | ImageNet | random | difference | only ImageNet succeeds | only random succeeds | sign test p |
|---|---|---|---|---|---|---|---|---|---|
| rgb | 330, fully visible | shows fully: 0, 2, 3, 6 | 480 | 0.521 | 0.354 | +0.167 | 137 | 57 | 8.8e-09 |
| rgb | 330, fully visible | shows faintly: 7, 10 | 240 | 0.221 | 0.138 | +0.083 | 42 | 22 | 0.017 |
| rgb | 430, fully or faintly | shows fully: 0, 2, 3, 6 | 480 | 0.546 | 0.312 | +0.233 | 156 | 44 | 6.8e-16 |
| rgb | 430, fully or faintly | shows faintly: 7, 10 | 240 | 0.350 | 0.267 | +0.083 | 43 | 23 | 0.019 |
| depth | 330, fully visible | shows fully: 0, 1, 3, 5 | 480 | 0.863 | 0.748 | +0.115 | 80 | 25 | 6.9e-08 |
| depth | 330, fully visible | shows faintly: 8, 10 | 240 | 0.479 | 0.296 | +0.183 | 58 | 14 | 1.6e-07 |
| depth | 430, fully or faintly | shows fully: 0, 1, 3, 5 | 480 | 0.848 | 0.652 | +0.196 | 113 | 19 | 1.8e-17 |
| depth | 430, fully or faintly | shows faintly: 8, 10 | 240 | 0.729 | 0.467 | +0.262 | 75 | 12 | 2.7e-12 |
| thermal | 330, fully visible | shows fully: 0, 1, 2, 4 | 480 | 0.577 | 0.379 | +0.198 | 136 | 41 | 4.2e-13 |
| thermal | 330, fully visible | shows faintly: 9, 10 | 240 | 0.263 | 0.167 | +0.096 | 40 | 17 | 0.0032 |
| thermal | 430, fully or faintly | shows fully: 0, 1, 2, 4 | 480 | 0.558 | 0.210 | +0.348 | 195 | 28 | 5.5e-32 |
| thermal | 430, fully or faintly | shows faintly: 9, 10 | 240 | 0.358 | 0.138 | +0.221 | 69 | 16 | 5.2e-09 |

Read-out (2) by the rules fixed before the runs:

- **Every reference policy is higher with ImageNet encoders**, for the three sensors, both recipes, where the sensor
  shows the glass fully and where faintly (p from 0.019 to 5.5e-32).
- **The reference rises in every cell: mean 0.674 (before 0.509).** Cells without depth 0.492 (0.335), cells with
  depth 0.826 (0.654).
- **All eleven cells are now shown to be solvable** (reference at least 0.30): cell 7 (rgb faint) 0.333 and cell 9
  (thermal faint) 0.383, the two that were below it. No single-sensor cell is below 0.15.

Observations outside the read-out:

- **With ImageNet encoders the faint demos no longer cost success where the sensor is clear**, and they are still
  needed where it is faint. The 430-demo policy against the 330-demo one, same episodes: where the sensor shows the
  glass fully rgb +0.025 (p = 0.29), depth -0.015 (p = 0.40), thermal -0.019 (p = 0.47); where faintly rgb +0.129
  (p = 4.7e-05), depth +0.250 (p = 2.7e-11), thermal +0.096 (p = 0.013). With random encoders the faint demos lowered
  depth and thermal where the sensor is clear.
- **The best baseline is below the reference in every cell again** (by 0.12 to 0.43; 0.428 against 0.674 on
  average), also where depth shows the glass (0.646 against 0.826). Seed 42 of the ImageNet rgbdt against the
  single-sensor ImageNet policy of a sensor that shows the glass, same episodes: cells 0, 1, 3, 5, depth on 330 demos
  0.863 against 0.706 (p = 3.2e-12); cells 8 and 10, depth on 430 demos 0.729 against 0.504 (p = 5.2e-09); cells 2
  and 4, thermal on 330 demos 0.583 against 0.208 (p = 1.8e-17); cells 2 and 6, rgb on 430 demos 0.567 against 0.192
  (p = 8e-20); cell 7, rgb on 430 demos 0.333 against 0.092 (p = 4.9e-06); cell 9, thermal on 430 demos 0.383
  against 0.050 (p = 2e-11). A policy with one image, trained on the demos where that sensor shows the glass, is above
  the policy with three images and all 700 demos in every group of cells.
- The reference stays the highest of several policies with 120 episodes per cell and one seed, so it leans high.

### Results of the baselines with ImageNet encoders (11/22 18:30; seeds 42, 43 and 44, 1320 test episodes per run)

Second part of the addition to stage 5. Mean over the three seeds and, in brackets, their range. Best baseline: the
highest mean of the cell. Reference: from the section above. Per run: `cells.json` in its folder.

| cell | rgb / depth / thermal | rgbdt | rgbd | rgbt | rgb | best baseline | reference |
|---|---|---|---|---|---|---|---|
| 0 | shows / shows / shows | 0.781 (0.775-0.792) | 0.725 (0.692-0.758) | 0.364 (0.342-0.400) | 0.244 (0.200-0.267) | 0.781 (rgbdt) | 0.900 |
| 1 | blind / shows / shows | 0.758 (0.750-0.775) | 0.686 (0.625-0.733) | 0.208 (0.200-0.217) | 0.047 (0.042-0.058) | 0.758 (rgbdt) | 0.875 |
| 2 | shows / blind / shows | 0.286 (0.225-0.333) | 0.178 (0.167-0.192) | 0.397 (0.392-0.408) | 0.311 (0.300-0.325) | 0.397 (rgbt) | 0.642 |
| 3 | shows / shows / blind | 0.692 (0.650-0.733) | 0.700 (0.683-0.717) | 0.208 (0.158-0.267) | 0.281 (0.242-0.350) | 0.700 (rgbd) | 0.867 |
| 4 | blind / blind / shows | 0.183 (0.175-0.192) | 0.050 (0.042-0.058) | 0.189 (0.158-0.242) | 0.039 (0.025-0.058) | 0.189 (rgbt) | 0.608 |
| 5 | blind / shows / blind | 0.658 (0.625-0.700) | 0.697 (0.633-0.742) | 0.047 (0.042-0.058) | 0.047 (0.042-0.058) | 0.697 (rgbd) | 0.858 |
| 6 | shows / blind / blind | 0.144 (0.108-0.167) | 0.169 (0.117-0.225) | 0.214 (0.183-0.242) | 0.256 (0.217-0.275) | 0.256 (rgb) | 0.492 |
| 7 | faint / blind / blind | 0.131 (0.092-0.150) | 0.136 (0.108-0.183) | 0.156 (0.142-0.175) | 0.181 (0.150-0.225) | 0.181 (rgb) | 0.333 |
| 8 | blind / faint / blind | 0.486 (0.458-0.525) | 0.481 (0.458-0.500) | 0.044 (0.025-0.058) | 0.053 (0.042-0.067) | 0.486 (rgbdt) | 0.742 |
| 9 | blind / blind / faint | 0.089 (0.050-0.117) | 0.064 (0.033-0.092) | 0.094 (0.067-0.108) | 0.056 (0.050-0.058) | 0.094 (rgbt) | 0.383 |
| 10 | faint / faint / faint | 0.503 (0.467-0.558) | 0.481 (0.450-0.508) | 0.217 (0.208-0.233) | 0.167 (0.158-0.183) | 0.503 (rgbdt) | 0.717 |
| | **macro average, eleven cells** | **0.428** (0.414-0.440) | **0.397** (0.387-0.404) | **0.194** (0.190-0.198) | **0.153** (0.140-0.161) | 0.458 | 0.674 |
| | cells without depth (2, 4, 6, 7, 9) | 0.167 | 0.119 | 0.210 | 0.168 | 0.223 | 0.492 |
| | cells with depth (0, 1, 3, 5, 8, 10) | 0.646 | 0.628 | 0.181 | 0.140 | 0.654 | 0.826 |
| | cells 0-6 | 0.500 | 0.458 | 0.233 | 0.175 | 0.540 | 0.749 |
| | cells 7-10 | 0.302 | 0.290 | 0.128 | 0.114 | 0.316 | 0.544 |
| | macro average, seeds 42 / 43 / 44 | 0.414 / 0.440 / 0.431 | 0.404 / 0.400 / 0.387 | 0.195 / 0.198 / 0.190 | 0.158 / 0.140 / 0.161 | | |

Read-out (1) by the rules fixed before the runs:

- **Every baseline is higher with ImageNet encoders than with random ones**, pooled on the same episodes: rgbdt
  0.428 against 0.293, rgbd 0.397 against 0.309, rgbt 0.194 against 0.088, rgb 0.153 against 0.068 (p from 2.2e-28
  to 1.8e-57). The macro average moves by at most 0.026 between seeds.
- **With ImageNet encoders rgbdt is above rgbd, so the thermal image helps the plain policy once the encoders are
  pretrained.** 0.428 against 0.397; only rgbdt succeeds in 545 episodes, only rgbd in 421, p = 7.4e-05. The
  stage-2 finding (rgbdt not above rgbd) was for random encoders and does not hold here. The difference is in the
  four cells where thermal shows the glass fully: cells 0, 1, 2 and 4 (+0.056, +0.072, +0.108, +0.133; p = 0.022,
  0.0088, 0.00021, 2.4e-09); none in the other seven.
- **No baseline is the best everywhere.** Best of the cell: rgbdt in cells 0, 1, 8 and 10, rgbd in 3 and 5, rgbt in
  2, 4 and 9, rgb in 6 and 7. Mean of the best baselines 0.458; the best single baseline is rgbdt with 0.428.
- **rgbdt is below a policy with fewer sensors where the extra sensors do not show the glass.** Cell 2 (rgb and
  thermal): below rgbt, -0.111, p = 0.00026. Cell 6 (rgb only): below rgbt, -0.069, p = 0.0052, and below rgb,
  -0.111, p = 5.5e-05. Cell 7 (rgb faint): below rgb, -0.050, p = 0.047. Likewise rgbd is below rgbt in cells 2 and
  4 (-0.219 and -0.139). Against rgbt and rgb overall rgbdt is far above (0.428 against 0.194 and 0.153), from the
  cells where depth shows the glass.

Read-out (3): the best baseline of each cell against the reference, and its seed-42 run against the reference policy
of that cell on the same 120 episodes.

| cell | best baseline (three seeds) | reference | seed 42 of that baseline | reference policy | only the baseline succeeds | only the reference succeeds | sign test p |
|---|---|---|---|---|---|---|---|
| 0 | 0.781 (rgbdt) | 0.900 | 0.775 | depth, 430 demos | 6 | 21 | 0.0059 |
| 1 | 0.758 (rgbdt) | 0.875 | 0.750 | depth, 430 demos | 5 | 20 | 0.0041 |
| 2 | 0.397 (rgbt) | 0.642 | 0.408 | rgb, 430 demos | 7 | 35 | 1.5e-05 |
| 3 | 0.700 (rgbd) | 0.867 | 0.700 | depth, 330 demos | 1 | 21 | 1.1e-05 |
| 4 | 0.189 (rgbt) | 0.608 | 0.167 | thermal, 330 demos | 3 | 56 | 1.2e-13 |
| 5 | 0.697 (rgbd) | 0.858 | 0.633 | depth, 330 demos | 8 | 35 | 4.2e-05 |
| 6 | 0.256 (rgb) | 0.492 | 0.275 | rgb, 430 demos | 11 | 37 | 0.00022 |
| 7 | 0.181 (rgb) | 0.333 | 0.167 | rgb, 430 demos | 9 | 29 | 0.0017 |
| 8 | 0.486 (rgbdt) | 0.742 | 0.525 | depth, 430 demos | 9 | 35 | 0.00011 |
| 9 | 0.094 (rgbt) | 0.383 | 0.108 | thermal, 430 demos | 6 | 39 | 5.4e-07 |
| 10 | 0.503 (rgbdt) | 0.717 | 0.483 | depth, 430 demos | 8 | 36 | 2.5e-05 |

- **In every cell the best baseline is below the single-sensor reference policy** (p from 0.0059 to 1.2e-13; over
  the eleven cells only the baseline succeeds in 73 episodes, only the reference in 364). The gap is 0.12 to 0.42 per
  cell, 0.216 on average; the reference leans high (one seed, the highest of several policies), and the test above
  is on the reference policy's own cell, so it leans the same way.

### State after stage 5 (11/22 18:30; all runs with ImageNet encoders)

| | macro average | cells without depth | cells with depth | seeds |
|---|---|---|---|---|
| reference (best single-sensor policy of the cell; uses the cells) | 0.674 | 0.492 | 0.826 | 1 |
| best baseline of each cell | 0.458 | 0.223 | 0.654 | 3 |
| rgbdt | 0.428 | 0.167 | 0.646 | 3 |
| rgbd | 0.397 | 0.119 | 0.628 | 3 |
| rgbt | 0.194 | 0.210 | 0.181 | 3 |
| rgb | 0.153 | 0.168 | 0.140 | 3 |

### Stage 6: the benchmark, its baselines and the comparison rules, frozen (11/22 18:40)

**Benchmark.** Level `L0_vis` (`sim/configs/L0_vis.yaml`), eleven visibility cells. Training set: the 700 demos of
`results/data/sim_demos_L0_vis_700.zarr` (seeds 0 to 699; 140 of cell 0, 70 each of cells 1 to 3, 50 each of cells 4
to 10). Test sets: `j_cells11` and `k_cells11_heldout`, 660 episodes each, 60 per cell. Protocol: 101 epochs, the
last checkpoint, default settings, nothing selected on the test sets, seeds 42, 43 and 44. Headline number: the macro
average over the eleven cells (1320 episodes per run), always with the table per cell
(`python policy/summarize_cells.py <run folders> [--versus <run folders>]`).

**Baselines of the comparison** (rule 4): the plain Diffusion Policy with rgbdt, rgbd, rgbt and rgb, ImageNet
encoders (`policy.obs_encoder.rgb_model.weights=IMAGENET1K_V1`), three seeds each. Frozen numbers:

| cell | rgb / depth / thermal | best baseline | which | reference (not a baseline) |
|---|---|---|---|---|
| 0 | shows / shows / shows | 0.781 | rgbdt | 0.900 |
| 1 | blind / shows / shows | 0.758 | rgbdt | 0.875 |
| 2 | shows / blind / shows | 0.397 | rgbt | 0.642 |
| 3 | shows / shows / blind | 0.700 | rgbd | 0.867 |
| 4 | blind / blind / shows | 0.189 | rgbt | 0.608 |
| 5 | blind / shows / blind | 0.697 | rgbd | 0.858 |
| 6 | shows / blind / blind | 0.256 | rgb | 0.492 |
| 7 | faint / blind / blind | 0.181 | rgb | 0.333 |
| 8 | blind / faint / blind | 0.486 | rgbdt | 0.742 |
| 9 | blind / blind / faint | 0.094 | rgbt | 0.383 |
| 10 | faint / faint / faint | 0.503 | rgbdt | 0.717 |
| | **macro average** | 0.458 (best single baseline: rgbdt, 0.428) | | 0.674 |

**Reported, not in the comparison** (each is below the best baseline above in every cell): the four baselines with
random encoders (three seeds), rgbdt with sensor dropout (random encoders, three seeds), and the single-sensor
policies on all 700 demos with their combination (random encoders, one seed).

**Reference.** The highest success among the single-sensor policies whose sensor shows the glass in the cell,
trained on the demos where it does. It uses the cells and has one seed; it shows what is reachable in a cell and is
not something a method has to reach.

**Rules for a method** (fixed 11/21, unchanged): (1) the same information as the baselines: sensor images, joint
values and the demo actions; the arm's kinematics, the camera calibration and ImageNet weights are allowed to any
run; (2) default settings, nothing selected on the test sets; (3) three seeds, and two policies are compared on the
same episodes; (4) per cell against the best baseline of that cell in the table above; (5) "better in nearly every
case": no cell in which the method is below that best baseline with p < 0.05 (pooled sign test on the same
episodes), and a macro average above every baseline (0.428). Runs that use more than rule 1 allows, such as a
segmentation pre-training on the simulator's glass masks, are reported apart from this comparison, with a control
that has the same supervision ([`../fusion_vis/README.md`](../fusion_vis/README.md)).

**Open, not blocking.** The reference has one seed. Sensor dropout and the combination of single-sensor policies
were not rerun with ImageNet encoders.

### Where the baselines fail (11/22 03:50, updated 18:30 with all ImageNet runs; from the evaluation records, no new runs, outside the read-outs)

`policy/analyze_failures.py` on the `episodes.json` files of the runs so far; full tables with the failure types in
[`failure_analysis.txt`](failure_analysis.txt). Measured: the horizontal distance between gripper and glass when the
gripper closes for the last time (the grasp the outcome is judged from). This is an analysis after the fact, not a
pre-registered read-out.

Success against that distance, 52800 episodes of 22 policies pooled:

| distance (cm) | 0-0.5 | 0.5-1 | 1-1.5 | 1.5-2 | 2-3 | 3-4 | 4-8 | over 8 |
|---|---|---|---|---|---|---|---|---|
| success | 0.969 | 0.845 | 0.460 | 0.299 | 0.222 | 0.077 | 0.010 | 0.000 |

"Reached": closed within 4 cm of the glass. "Placed": within 1.5 cm.

| cells | policy | success | reached | placed | median distance (cm) | success when placed |
|---|---|---|---|---|---|---|
| depth shows fully (0, 1, 3, 5) | depth only, 330 fully visible demos | 0.748 | 0.950 | 0.810 | 0.7 | 0.889 |
| | rgbdt, ImageNet encoders (3 seeds) | 0.722 | 0.889 | 0.684 | 0.8 | 0.964 |
| | rgbd (3 seeds) | 0.581 | 0.974 | 0.692 | 1.0 | 0.757 |
| | rgbdt (3 seeds) | 0.547 | 0.958 | 0.640 | 1.2 | 0.746 |
| | depth only, all 700 demos | 0.421 | 0.915 | 0.517 | 1.5 | 0.706 |
| | rgbdt with sensor dropout (3 seeds) | 0.435 | 0.919 | 0.494 | 1.5 | 0.705 |
| no depth, thermal shows fully (2, 4) | thermal only, 330 fully visible demos | 0.379 | 0.796 | 0.433 | 1.8 | 0.692 |
| | rgbdt, ImageNet encoders (3 seeds) | 0.235 | 0.393 | 0.143 | 5.2 | 0.835 |
| | rgbt (3 seeds) | 0.107 | 0.399 | 0.118 | 4.9 | 0.529 |
| | rgbdt (3 seeds) | 0.096 | 0.286 | 0.076 | 6.3 | 0.673 |
| | depth only, 330 (blind in these cells) | 0.037 | 0.062 | 0.021 | 15.0 | - |
| no depth, rgb shows fully (2, 6) | rgb only, 330 fully visible demos | 0.358 | 0.842 | 0.396 | 1.9 | 0.716 |
| | rgbdt, ImageNet encoders (3 seeds) | 0.215 | 0.390 | 0.142 | 5.0 | 0.784 |
| | rgbt (3 seeds) | 0.154 | 0.615 | 0.188 | 3.4 | 0.533 |
| | rgbdt (3 seeds) | 0.117 | 0.369 | 0.104 | 5.2 | 0.560 |
| | depth only, 330 (blind in these cells) | 0.046 | 0.058 | 0.008 | 13.9 | - |

- **The task is decided by where the gripper closes, to about 1 cm.** 0.97 within 0.5 cm, 0.85 up to 1 cm, 0.46 up to
  1.5 cm, nothing beyond 4 cm. Once the gripper is placed within 1.5 cm the policies with random encoders succeed in
  0.53 to 0.89 of the episodes and the ImageNet rgbdt in 0.78 to 0.96; the larger differences are in how often a
  policy gets there.
- **Without depth the plain policies do not find the glass.** rgbdt closes within 4 cm of it in 0.29 of the episodes
  where thermal shows it and 0.37 where rgb shows it, typically 5 to 6 cm away, and "closed away from every object"
  is 0.30 of all episodes in the five cells without depth. The single-sensor policy of the sensor that shows the glass
  reaches it in 0.80 and 0.84, typically within 2 cm; a policy that is blind there in 0.06 (14 to 15 cm away). The
  plain rgbdt is between the two: it reads rgb and thermal a little. The ImageNet rgbdt reaches the glass hardly more
  often (0.39 in both groups, 5 cm away); its gain in these cells comes mostly from the grasps that are placed.
- **Where depth shows the glass the plain policies find it and lose on precision.** All policies with depth close
  within 4 cm in 0.92 to 0.98 of the episodes. Placed within 1.5 cm: 0.81 for the depth-only policy on the fully
  visible demos, 0.69 for rgbd, 0.64 for rgbdt, about 0.50 for the depth-only policy on all 700 demos and for sensor
  dropout. The failures there are missed grasps (0.26 to 0.28: closed near the glass, not lifted).
- **Rgb and thermal are coarser than depth even at their best.** Their single-sensor policies place the gripper
  within 1.5 cm in 0.40 and 0.43 of the episodes where the sensor shows the glass fully (depth: 0.81). This is what
  bounds the references of the cells without depth (0.38 to 0.42).

**With ImageNet encoders** (added 11/22 18:30, all runs of the addition to stage 5):

| cells | policy (ImageNet encoders) | success | reached | placed | median distance (cm) | success when placed |
|---|---|---|---|---|---|---|
| depth shows fully (0, 1, 3, 5) | depth only, 330 fully visible demos | 0.863 | 0.973 | 0.908 | 0.5 | 0.933 |
| | rgbdt (3 seeds) | 0.722 | 0.889 | 0.684 | 0.8 | 0.964 |
| | rgbd (3 seeds) | 0.702 | 0.951 | 0.780 | 0.8 | 0.858 |
| no depth, thermal shows fully (2, 4) | thermal only, 330 fully visible demos | 0.583 | 0.850 | 0.683 | 1.0 | 0.780 |
| | rgbt (3 seeds) | 0.293 | 0.564 | 0.274 | 3.1 | 0.777 |
| | rgbdt (3 seeds) | 0.235 | 0.393 | 0.143 | 5.2 | 0.835 |
| | rgbd (3 seeds) | 0.114 | 0.344 | 0.096 | 5.8 | 0.623 |
| no depth, rgb shows fully (2, 6) | rgb only, 430 demos | 0.567 | 0.879 | 0.629 | 1.1 | 0.775 |
| | rgb (3 seeds) | 0.283 | 0.757 | 0.304 | 2.3 | 0.644 |
| | rgbt (3 seeds) | 0.306 | 0.706 | 0.326 | 2.4 | 0.715 |
| | rgbd (3 seeds) | 0.174 | 0.469 | 0.136 | 4.3 | 0.612 |
| | rgbdt (3 seeds) | 0.215 | 0.390 | 0.142 | 5.0 | 0.784 |

- **The picture is the same with ImageNet encoders, at a higher level.** The single-sensor policies place the
  gripper within 1.5 cm in 0.91 (depth), 0.68 (thermal) and 0.63 (rgb) of the episodes where their sensor shows the
  glass fully (with random encoders 0.81, 0.43 and 0.40); the baselines trained on all demos in 0.68 to 0.78 where
  depth shows the glass and in 0.10 to 0.33 where it does not.
- **A sensor that does not show the glass makes the plain policy find it less often.** Where rgb shows the glass
  and depth does not (cells 2 and 6), the gripper closes within 4 cm of the glass in 0.76 of the episodes for rgb,
  0.71 for rgbt, 0.47 for rgbd and 0.39 for rgbdt.

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
