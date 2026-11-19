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

No policy has been trained or evaluated on this level. The first runs are listed [at the end](#proposed-first-runs-not-started)
and wait for approval.

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

## Proposed first runs (not started)

Waiting for approval; nothing below has been run.

1. **Baselines**: Diffusion Policy with `rgb`, `rgbd`, `rgbt`, `rgbdt` on the 500 demos, seed 42, 101 epochs, the
   settings of the `fusion_bench` baselines; rollouts during training on `h_cells` (35 episodes, monitoring only).
   Estimated at 6–7 h per training and 1 h per evaluation.
2. **Gate for three sensors**: the visibility gate so far decides depth and thermal only. It needs a third flag for
   rgb and the level's segmentation network (check 4). Its design is written up and approved before it is trained.

## Files

- `figure_cells.png`; `thermal_contrast_sweep.{txt,json}`; `real_thermal_contrast.{txt,json}`; `teacher_check.txt`;
  `cells_check_L0_bench_net.{txt,json}`.
- Not in git: the demo set `results/data/sim_demos_L0_grid_500.zarr`; the segmentation frames, network and logs
  under `data/grid_bench/`.
