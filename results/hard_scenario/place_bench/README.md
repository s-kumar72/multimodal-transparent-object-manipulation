# Pick-and-place on the visibility benchmark: level `L0_vis_place`

Started 11/24 (decision of the user: a harder task than the pick). This file defines the level and records how it was
checked, and lists the runs on it.

## Why

On `L0_vis` the task is to pick a glass that stands still and lift it 10 cm. A policy that is given the grasp point
once and no images (`hcg_blind`, [`../cue_gate/README.md`](../cue_gate/README.md)) succeeds as often as one that also
sees the images: after the glass has been found, nothing else has to be seen. Pick-and-place adds a second place that
has to be found, a longer episode (about twice the steps) and a phase in which the glass is carried.

What it does not add: the scene still does not change while the arm moves, so in principle both places can be read
from the first frame. Whether the images matter during the motion is a question for the runs, not something the level
guarantees.

## The level

`sim/configs/L0_vis_place.yaml` (`base: L0_vis.yaml`, `task: place`, meta/ep_level id 36; in
`sim/configs/manifest.json`). ![teacher episodes](figure_place.png)

- **Scene.** The scenes of `L0_vis`, seed for seed: the same glass (tumbler, bottle, wineglass, jar, flask), position,
  visibility cell (eleven cells, same weights), table, light and one to three opaque distractors. One thing is added:
  a coaster. The test sets are the existing ones with the new level, `test_set("L0_vis_place", "j_cells11")` and
  `"k_cells11_heldout"` (660 episodes each, 60 per cell, the same seeds as for `L0_vis`).
- **Coaster.** An opaque disc 12 cm wide and 6 mm thick, fixed to the table (it cannot be pushed away), always with
  the same look: dark with a light centre. Its temperature is within 1 degC of the room. So the colour camera shows
  it in every cell, also in the cells in which the colour camera does not show the glass (there the glass is clear;
  the light is normal); depth (6 mm of height) and thermal hardly show it. *Measured 11/25 (step 3, perception): a
  network trained to find the place pixel finds it from depth in 0.96 of the frames and from thermal in 0.66, so
  "hardly" holds for neither; the coaster is shown by all three sensors, best by colour and depth.*
- **Where the coaster is.** A random spot on the near part of the table (x 0.27 to 0.42 m, y -0.28 to 0.28 m, no
  further from the arm's base than the far corners of the region the glass spawns in, 0.446 m, where the top-down IK
  still reaches the carry height), at least 2 cm (edge to edge) from the glass and from every distractor, and such
  that the glass can be carried to it in a straight line without passing over a distractor. It has its own random
  stream and is drawn last, so nothing else in the scene depends on it. Distance from the glass at the start: 11 to
  42 cm, median 15 cm (17 cm in `k_cells11_heldout`).
- **Task and success.** Pick the glass, carry it to the coaster, put it down and let go. Success
  (`GlassPickEnv.placed`), all at once: the glass stands on the coaster's top (within 5 mm in height), upright (leaning
  less than 10 degrees), its footprint entirely inside the coaster (its centre within 2.5 to 3 cm of the coaster's
  centre, depending on the object), at rest, and the tool is 5 cm above the glass's grasp point. The episode ends
  there. For the policy the episode is at most 240 steps (`task.env_runner.max_steps=240`; the pick uses 120).
- **Teacher.** The pick of the pick levels, unchanged (the same random draws), then: carry at the lift height in a
  straight line (about 1 cm per step), down over 12 to 17 steps until the base is 3 mm above the coaster, open for 6
  steps, back up in 12. Episodes are 99 to 142 steps, median 114 (the pick: 55 to 71).
- **Contacts.** The scene of this level (`sim/assets/gen3_lite/place_scene.xml`) uses elliptic friction cones with
  impratio 10. With the settings of the pick levels a held glass creeps down between the fingers while it is carried
  (the tumbler by about 3 cm, median; it then hit the coaster early and fell over in 14 of 60 episodes, the wineglass
  in 2 of 60); with these it stays within 2 mm. The 10 cm lift of the pick levels never showed this. The pick levels are not changed.
- **Failure labels** of an evaluation episode (`policy/glass_gym_env.py`): the labels of the pick when the glass was
  never lifted 10 cm; after that `place_fell`, `place_held`, `place_off`, `place_other`. The episode record also has
  `picked`, `place_offset` (m, at the end) and `coaster_xy`.

## Checks (11/24)

- **Existing levels are unchanged.** `python sim/regression_test.py --check` against arrays saved before the change,
  for the default scene and for `L0_vis` (`--config L0_vis.yaml`): 190 of 190 arrays identical in both (first frames
  of 20 seeds and whole teacher episodes of 5).
- **The teacher solves it.** [`teacher_check.txt`](teacher_check.txt) (`python sim/place_check.py --n 700 --test_set
  j_cells11 --n_test 660`, and the same with `k_cells11_heldout`): 700 of 700 training seeds, 660 of 660 on each test
  set, every object and every cell; no distractor moved. The glass ends 0.2 cm (median; at most 0.4 cm) from the
  coaster's centre.
- **Rare cases of the coaster's position**, seeds 0 to 2999: one seed in which the straight path is not clear, one
  with no free spot at all (the coaster then overlaps a distractor; the teacher fails and the seed is not a demo).
  Test sets: one episode of `j_cells11` without a clear path (the teacher succeeds), none without a free spot.
- **The policy-side env** (`GlassPickGymEnv`, level `L0_vis_place`): with the teacher's actions reward 1 after 106 and
  107 steps; an episode that never lets go is labelled `place_held`, one that puts the glass down 8 cm off
  `place_off`, one that does nothing `timeout`.

## Choices made here that can still be changed (nothing has been trained)

1. The coaster always looks the same and only the colour camera shows it. With a random colour it was hard to see on
   the table when the colours were close, which would have been a difficulty that has nothing to do with glass.
2. Success needs the whole footprint on the coaster (2.5 to 3 cm of tolerance). The grasp needs about 1 cm, so the
   pick stays the harder part.
3. The coaster is fixed to the table and thin; a thicker one would also show in depth.
4. The straight path to the coaster is kept clear, so no obstacle has to be avoided.

## Runs, steps 1 and 2: demos and baselines (approved by the user 11/24 04:33 KST; listed 04:38 KST, before any of them starts)

**Step 1, training demos.** 700 demos of the scripted teacher from seed 0 (`python sim/collect_demos.py --n 700
--config sim/configs/L0_vis_place.yaml --out results/data/sim_demos_L0_vis_place_700.zarr`); failed teacher episodes
are skipped, as always. About 1.8 times the frames of the pick set.

**Step 2, baselines.** The protocol of the frozen baselines of `L0_vis` (Stage 6 of
[`../grid_bench/README.md`](../grid_bench/README.md)): plain Diffusion Policy (`policy/train.sh`), ImageNet encoders,
101 epochs, last checkpoint, rollouts during training on `j_cells11` (55 episodes), final evaluation on `j_cells11`
and `k_cells11_heldout` (660 episodes each, 120 per cell over both), seed 42. The only difference: episodes of at
most 240 steps (`task.env_runner.max_steps=240`).

| run | inputs |
|---|---|
| `L0_vis_place_rgbdt_700demos_imagenet` | colour, depth, thermal, joints |
| `L0_vis_place_rgbd_700demos_imagenet` | colour, depth, joints |
| `L0_vis_place_rgbt_700demos_imagenet` | colour, thermal, joints |
| `L0_vis_place_rgb_700demos_imagenet` | colour, joints |

Command of a run (queue `data/runs/fv_queue.sh` of the fusion-vis worktree, behind the lines it already has; results
in `results/hard_scenario/place_bench/<run>/`):
`GROUP=place_bench TOPK=0 NTEST=55 NEVAL=660 TEST_SETS="j_cells11 k_cells11_heldout" TAG=imagenet
EXTRA="training.seed=42 task.env_runner.test_set=j_cells11 task.env_runner.max_steps=240
policy.obs_encoder.rgb_model.weights=IMAGENET1K_V1" bash scripts/run_experiment.sh L0_vis_place <inputs> 700`.
A run is expected to take about twice a pick run (7 to 8 hours). A run that fails or is stopped is reported as such
and not started again (a resumed run would train past epoch 100).

**Step 1, done (11/24 04:43 KST).** `results/data/sim_demos_L0_vis_place_700.zarr` (not in git): 700 demos from seeds
0 to 699, no teacher failure, 140 per object, the cells in the weights of `L0_vis` (140, 3 x 70, 7 x 50). 80538
frames, 100 to 139 steps per demo (median 114): 1.85 times the 43489 frames of the pick set.

**Step 2, queued (11/24 05:00 KST)** after a short end-to-end test (two debug epochs with rollouts of 240 steps and
the final evaluation, deleted afterwards): it ran through, and with ten envs it took about 15 to 20 GB of RAM (a pick
run: 14 to 16 GB). An epoch is 1124 batches (the pick: 574), so a run is expected to take about 8 hours. The four
runs are the last lines of the queue; the first is expected to start about 09:30 KST.

**Read-out, fixed now** (per run, from `episodes.json` of the two final evaluations, 1320 episodes):

1. Success (the glass placed) over all episodes, per test set and per cell, with the macro average over the cells.
2. The pick and the place apart: the share of episodes in which the glass was picked (lifted 10 cm at some point),
   and the success among those.
3. The failure labels.
4. Next to it, not a test: the success of the same inputs on the pick level `L0_vis` (the frozen run of seed 42, the
   same scenes and seeds), per cell, against "picked" here. It shows whether learning the place costs the pick.
5. The best baseline of each cell, as in Stage 6. One seed is a first reading; seeds 43 and 44 are not approved yet.

Not in this step: single-sensor references that are told the cell, sensor dropout, and any method. Results are
reported as they come out, whatever they are.

**Later, not specified yet.** The method (hindsight cue gating) on this level: the cue of the pick as it is, and
whether the place gets a cue of its own (the pixel where the gripper opens in the demos) is the open design question.

## Results of step 2: the four baselines (11/24 22:25 KST; seed 42, 1320 test episodes per run, 120 per cell)

All four runs trained to epoch 100 and were evaluated; none failed or was stopped. They started at 08:00 (rgbdt),
09:50 (rgbd), 11:01 (rgbt) and 15:24 KST (rgb) and were done, with the final evaluation, at 15:24, 15:40, 16:44 and
19:48 KST (4.4 to 7.4 hours, depending on how many runs shared the machine). Rollouts during training (55 episodes of
`j_cells11`, epochs 25 / 50 / 75 / 100): rgbdt 0.40 / 0.51 / 0.53 / 0.60, rgbd 0.42 / 0.55 / 0.55 / 0.64, rgbt 0.16 /
0.15 / 0.35 / 0.31, rgb 0.02 / 0.07 / 0.24 / 0.11. The read-out is [`readout_baselines.txt`](readout_baselines.txt)
(`python policy/readout_place.py`).

**1. Success (the glass placed).**

| cell | sensors that show the glass | rgbdt | rgbd | rgbt | rgb |
|---|---|---|---|---|---|
| 0 | rgb+depth+thermal | 0.950 | 0.925 | 0.458 | 0.125 |
| 1 | depth+thermal | 0.883 | 0.933 | 0.417 | 0.075 |
| 2 | rgb+thermal | 0.267 | 0.108 | 0.475 | 0.133 |
| 3 | rgb+depth | 0.933 | 0.975 | 0.092 | 0.142 |
| 4 | thermal | 0.200 | 0.050 | 0.483 | 0.050 |
| 5 | depth | 0.900 | 0.942 | 0.075 | 0.092 |
| 6 | rgb | 0.142 | 0.100 | 0.117 | 0.133 |
| 7 | rgb (faint) | 0.058 | 0.083 | 0.108 | 0.100 |
| 8 | depth (faint) | 0.700 | 0.842 | 0.075 | 0.042 |
| 9 | thermal (faint) | 0.208 | 0.108 | 0.300 | 0.067 |
| 10 | all three (faint) | 0.658 | 0.817 | 0.308 | 0.100 |
| all (macro average over the cells) | | 0.536 | 0.535 | 0.264 | 0.096 |
| `j_cells11` / `k_cells11_heldout` | | 0.555 / 0.518 | 0.542 / 0.527 | 0.280 / 0.248 | 0.092 / 0.100 |

- With depth in the inputs, the cells in which depth shows the glass fully (0, 1, 3, 5) are at 0.88 to 0.98 and the
  ones in which it shows it faintly (8, 10) at 0.66 to 0.84. Every cell without depth is at 0.27 or below for them.
- rgbdt and rgbd are equal over all episodes but not per cell: rgbd is higher where depth is faint (cells 8 and 10:
  0.842 and 0.817 against 0.700 and 0.658), rgbdt where thermal is the sensor that shows the glass (cells 2, 4, 9:
  0.267, 0.200, 0.208 against 0.108, 0.050, 0.108).
- rgbt is the highest in the thermal cells (2, 4, 9: 0.475, 0.483, 0.300). rgb is at 0.096 and above 0.142 nowhere.
- The cells in which only the colour camera shows the glass (6, 7) are at 0.06 to 0.14 for every baseline.

**2. The pick and the place apart.**

| | rgbdt | rgbd | rgbt | rgb |
|---|---|---|---|---|
| glass picked (lifted 10 cm at some point), share of all episodes | 0.605 | 0.574 | 0.342 | 0.124 |
| placed, among the picked | 0.886 | 0.926 | 0.757 | 0.701 |
| never picked / picked and not placed, share of all episodes | 0.395 / 0.069 | 0.426 / 0.042 | 0.658 / 0.083 | 0.876 / 0.037 |

Most failures are before the glass is lifted. Where depth shows the glass fully, rgbdt and rgbd pick it in 0.95 to
0.99 of the episodes and then place it in 0.93 to 0.99 of those (per cell in the read-out). The runs without depth
also place less often once the glass is picked (0.757, 0.701), although the colour camera shows the coaster to every
run in every cell; why was not examined.

**3. Failure labels** (share of all episodes; rgbdt / rgbd / rgbt / rgb). Before the pick: `missed_grasp` 0.167 /
0.156 / 0.270 / 0.384, `other` 0.141 / 0.161 / 0.228 / 0.280, `dropped` 0.055 / 0.057 / 0.091 / 0.106, `wrong_height`
0.023 / 0.037 / 0.042 / 0.086, `wrong_object` 0.017 or less. After the pick: `place_fell` 0.050 / 0.030 / 0.064 /
0.030, `place_off` 0.018 / 0.011 / 0.019 / 0.006, `place_held` and `place_other` 0.001 or less. `timeout` is 0.005 to
0.008: the 240 steps are enough.

**4. Against the pick level (next to the read-out, not a test).** The same inputs on `L0_vis` (frozen run of seed 42,
the same scenes and seeds), success there against "picked" here, on the same episodes:

| inputs | cells | success on the pick level | picked here | only the pick level / only here | p |
|---|---|---|---|---|---|
| rgbdt | all | 0.414 | 0.605 | 62 / 314 | 1.1e-41 |
| rgbdt | depth shows the glass at least faintly (0, 1, 3, 5, 8, 10) | 0.639 | 0.917 | 11 / 211 | 3.9e-49 |
| rgbdt | it does not (2, 4, 6, 7, 9) | 0.143 | 0.230 | 51 / 103 | 3.4e-05 |
| rgbd | all | 0.404 | 0.574 | 75 / 300 | 5.6e-33 |
| rgbd | depth cells | 0.621 | 0.949 | 5 / 241 | 1.3e-64 |
| rgbd | the others | 0.143 | 0.125 | 70 / 59 | 0.38 |
| rgbt | all | 0.195 | 0.342 | 121 / 315 | 5.3e-21 |
| rgb | all | 0.158 | 0.124 | 174 / 130 | 0.014 |

Learning the place did not cost the pick. With depth or thermal in the inputs the glass is picked more often here
than the pick succeeds on the pick level, and by a wide margin; with colour alone it is picked somewhat less often.
So for these baselines the level is not harder than the pick level in terms of success: rgbdt places the glass in
0.536 of the episodes here and picks it in 0.414 there.

The two sides of this comparison differ in more than the task, and the files cannot tell these apart: (a) the
contact settings (elliptic cones with impratio 10 here, the defaults on the pick levels; see "Contacts" above), (b)
the gradient steps (the same 101 epochs, but 1124 batches per epoch here against 574), (c) the step limit (240
against 120), (d) the coaster in the images and the longer demos.

A further look from the same files, not in the fixed read-out and not a test ([`pick_compare.txt`](pick_compare.txt),
`python policy/place_pick_compare.py`; distances in the table plane between the tool and the glass's grasp point at
the first close of the gripper):

| inputs | level | picked | closes per episode | picked with a single close | first close within 1.5 cm | picked when the first close is 0 to 1 cm away | 1 to 1.5 cm | 1.5 to 3 cm | over 3 cm |
|---|---|---|---|---|---|---|---|---|---|
| rgbdt | pick | 0.414 | 1.97 | 0.365 | 0.465 | 0.873 | 0.480 | 0.293 | 0.033 |
| rgbdt | place | 0.605 | 1.19 | 0.581 | 0.479 | 0.986 | 0.973 | 0.685 | 0.040 |
| rgbd | pick | 0.404 | 1.55 | 0.377 | 0.469 | 0.878 | 0.451 | 0.223 | 0.030 |
| rgbd | place | 0.574 | 1.14 | 0.562 | 0.499 | 0.996 | 0.959 | 0.610 | 0.016 |
| rgbt | pick | 0.195 | 1.98 | 0.157 | 0.195 | 0.772 | 0.430 | 0.240 | 0.021 |
| rgbt | place | 0.342 | 1.26 | 0.308 | 0.182 | 0.934 | 0.891 | 0.604 | 0.044 |
| rgb | pick | 0.158 | 1.24 | 0.133 | 0.193 | 0.651 | 0.372 | 0.201 | 0.020 |
| rgb | place | 0.124 | 1.39 | 0.078 | 0.072 | 0.595 | 0.585 | 0.350 | 0.026 |

- **The runs here do not find the glass better.** With depth or thermal in the inputs the first close is as near to
  the glass on both levels (within 1.5 cm: 0.465 and 0.479 for rgbdt, 0.469 and 0.499 for rgbd, 0.195 and 0.182 for
  rgbt).
- **What differs is what follows a first close that is near.** On the pick level a first close 1 to 1.5 cm from the
  glass ends in a pick in 0.43 to 0.48 of the episodes, here in 0.89 to 0.97; at 1.5 to 3 cm, 0.22 to 0.29 against
  0.60 to 0.69. On the same episodes with the first close within 1.5 cm on both levels (491 for rgbdt, 516 for rgbd)
  the glass is picked only here in 101 and 112 and only on the pick level in 0 and 2 (p = 8e-31 and 6e-31); on the
  pick level those end as `dropped`, `missed_grasp` or `wrong_height`.
- **It is not the extra time.** The runs here close the gripper fewer times per episode (1.19 against 1.97 for
  rgbdt), and the share picked with a single close is 0.581 against 0.365.
- **It depends on the object.** After a first close within 1.5 cm the jar is picked on both levels (0.98 and 0.94
  for rgbdt); the bottle in 0.57 on the pick level and 1.00 here, the flask 0.65 and 0.98, the tumbler 0.73 and
  0.99, the wineglass 0.81 and 1.00.
- **Colour alone goes the other way** for another reason: its first close is further from the glass here (within
  1.5 cm in 0.072 of the episodes against 0.193).

What this suggests and does not show: on the pick level a grasp that is at the glass often does not hold, and here it
does. Of the differences (a) to (d), the contact settings are the one that acts at that point, and they were changed
for this level because a held glass crept between the fingers with the settings of the pick levels. But the policies
differ too (twice the gradient steps, longer demos), so this is not separated. A check that would separate it: the
final evaluation of the frozen pick-level checkpoints in the scenes of `L0_vis` with the contact settings of this
level (evaluation only; a new config, the existing levels unchanged). Not run and not approved.

If it is the contacts, it concerns the pick benchmark and not only this level (last table of
[`pick_compare.txt`](pick_compare.txt), seed 42): on `L0_vis` the rgbdt baseline fails after a first close within
1.5 cm in 0.117 of all episodes (0.20 of its failed episodes), and the method `hcg_cue` in 0.094 (0.46 of its failed
episodes: `dropped` 55, `missed_grasp` 54, `wrong_height` 15). The earlier reading that a pick needs the gripper
within about 1 cm of the glass ([`../grid_bench/README.md`](../grid_bench/README.md), failure analysis) would then be
a property of those contact settings: here a first close 1 to 1.5 cm away is picked in 0.97.

**5. The best baseline of each cell** (one seed, a first reading): rgbdt in cells 0 and 6, rgbd in 1, 3, 5, 8 and 10,
rgbt in 2, 4, 7 and 9. The mean of the best per cell is 0.633, against 0.536 for the best single baseline. In cells
0, 6 and 7 the first two are within 0.03 of each other, which 120 episodes of one seed do not order.

**What step 2 says (one seed each).**

1. The structure of the pick benchmark carries over to this level: high where depth shows the glass, low elsewhere,
   each sensor's baseline is the best in its own cells and none is the best everywhere, and no baseline solves the
   cells in which only the colour camera shows the glass.
2. The place itself is a small part of the difficulty for the baselines that have depth (0.89 to 0.93 of the picked
   glasses are placed); the pick is where the episodes are lost, as on the pick level.
3. The level was built as a harder task than the pick. For the baselines it is not harder in success, because the
   pick succeeds more often here than on the pick level; the further look above points at the contact settings, which
   is not confirmed.
4. Room for a method on this level is in the cells without depth (best baseline 0.11 to 0.48 in cells 2, 4, 6, 7, 9)
   and in the faint ones (0.82 and 0.84); the cells in which depth shows the glass fully are at 0.93 to 0.98.

Not approved and not started: seeds 43 and 44, the references, the method on this level, and the check of the
contact settings.

## Step 3: hindsight cue gating on this level (approved by the user 11/25; listed 11/25 01:32 KST, before anything is fitted or trained)

The method of [`../cue_gate/README.md`](../cue_gate/README.md), with one addition for the second place that has to be
found. Nothing about the method is chosen on this level: the networks, the gate and the policy arms are those of the
pick level, with their settings.

**Two cues.** A pick-and-place demo says two things in hindsight: where the gripper closed (the glass) and where it
opened again (the place).

- *Grasp cue*, as on the pick level: the tool position at the first step that closes the gripper, per-sensor maps
  from the first four frames, a verifier per sensor, the trusted sensors that agree fused
  (`python policy/hindsight_cue.py results/data/sim_demos_L0_vis_place_700.zarr --level L0_vis_place --out
  data/hcue/L0_vis_place_700`).
- *Place cue*, new (`--event release --out data/hcue/L0_vis_place_700_release`): the label is the point of the table
  plane under the tool at the first step that opens the gripper again. The table plane and not the tool itself,
  because the tool is 8 to 16 cm above the table at that step, depending on the glass, so its pixel is not where the
  place target is seen. Same networks, same verifier, same fusion rule; the cue vector has the same nine entries
  (point, pixel, confidence, r_rgb, r_depth, r_thermal) and says where the glass will be put down and which sensors
  show that place. In the demos the tool is within 0.1 cm of the coaster's centre at that step.
- The policy gets both, one after the other, in the observation `hcue` (18 entries; task `glass_pick_hcue2`,
  `CUES=2` of `policy/train_cuegate.sh`). With the gate on, a sensor's feature is switched off when the sensor is
  trusted for neither target (the larger of its two r is below 0.5).

**Information used.** As on the pick level (sensor images, joint values, the demonstrated actions, arm kinematics,
the camera calibration, ImageNet weights), plus one constant of the setup: the height of the table in the robot
frame. The coaster's position, the cells and the glass's position appear in the reports only.

**What the level asks of the gate that the pick level does not.** The colour camera shows the coaster in every cell;
in cells 1, 4, 5, 8 and 9 it does not show the glass. There the two targets are shown by different sensors, so a rule
that switches the colour camera off (as the gate of the pick level does in those cells) would hide the coaster.

**Perception, reported before the policies (no choice depends on it).** For both cues: the report of
`hindsight_cue.py` on the training demos, out of fold, and `python policy/check_cue_env.py <cue.pt> --level
L0_vis_place --test_set <set>` on both test sets (660 episodes each; the place cue against the coaster's centre), in
`perception/`.

**Perception: results (11/25 02:10 KST, before any policy run starts).** Records in [`perception/`](perception/):
`grasp_cue_report.txt`, `place_cue_report.txt` (training demos, out of fold), `env_<grasp | place>_<test set>.txt`
(the cue the evaluation env computes at reset, 660 episodes per test set).

| cue | training demos, out of fold: within 1.5 px / within 1.5 cm | `j_cells11`: within 1 / 1.5 / 4 cm | `k_cells11_heldout`: within 1 / 1.5 / 4 cm | gate, median: rgb / depth / thermal (`j_cells11`) | sensors trusted per episode |
|---|---|---|---|---|---|
| grasp (the glass) | 0.95 / 0.90 | 0.82 / 0.91 / 0.98 | 0.76 / 0.85 / 0.92 | by cell, as on the pick level | 1.79 |
| place (the coaster) | 1.00 / 0.96 | 0.94 / 0.99 / 1.00 | 0.92 / 0.97 / 1.00 | 1.00 / 1.00 / 0.93 | 2.70 |

- The grasp cue is as on the pick level (there 0.93 and 0.88 within 1.5 cm on the two test sets); its weak cells are
  again 8 and 9 (0.78 and 0.62 on `j_cells11`).
- The place cue is right in nearly every episode of every cell.
- **The coaster is not seen by the colour camera alone.** The level's description says that depth and thermal hardly
  show it. The cue networks say otherwise: the depth network's own peak is within 2 px of the place pixel in 0.96 of
  the training frames (the 6 mm of height are enough) and the thermal network's in 0.66, and their gates are 1.00
  and about 0.9. So with the rule fixed above (a sensor is on when it is trusted for either target) the colour camera
  and depth are on in every episode and thermal in most: the gate of `hcg2_full` switches off little on this level,
  and "the two targets are shown by different sensors" holds only for thermal. The rule and the runs stay as
  listed; `hcg2_full` against `hcg2_cue` is read with this in mind.

**Policy runs** (seed 42; the protocol of the baselines of this level: 700 demos, 101 epochs, last checkpoint,
ImageNet encoders, episodes of at most 240 steps, rollouts during training on `j_cells11` with 55 episodes, final
evaluation on `j_cells11` and `k_cells11_heldout` with 660 episodes each; queue `data/runs/fv_queue.sh` of the
fusion-vis worktree, results in `results/hard_scenario/place_bench/<run>/`):

| run | arm | cues | images | features gated | question |
|---|---|---|---|---|---|
| `L0_vis_place_rgbdt_700demos_hcg2_full` | `full` | grasp + place | yes | yes | the method |
| `L0_vis_place_rgbdt_700demos_hcg2_cue` | `cue` | grasp + place | yes | no | does gating the features add to the cues? |
| `L0_vis_place_rgbdt_700demos_hcg_cue` | `cue` | grasp only | yes | no | does the place need a cue of its own? |
| `L0_vis_place_rgbdt_700demos_hcg2_blind` | `blind` | grasp + place | none | - | do the images add to the cues and the joints on this task? |

Command: `[CUES=2] GROUP=place_bench TOPK=0 NTEST=55 NEVAL=660 TEST_SETS="j_cells11 k_cells11_heldout" TAG=<hcg2_full
| hcg2_cue | hcg_cue | hcg2_blind> TRAIN="policy/train_cuegate.sh <arm>" EXTRA="training.seed=42
task.env_runner.test_set=j_cells11 task.env_runner.max_steps=240 [task.hcue_dir=data/hcue/L0_vis_place_700]" bash
scripts/run_experiment.sh L0_vis_place rgbdt 700` (`CUES=2` for the three runs with both cues; `task.hcue_dir` for
the run with the grasp cue only). A run that fails or is stopped is reported as such and not started again.

**Read-out, fixed now** (per run, 1320 episodes; `python policy/readout_place_method.py`):

1. Against the baselines: success against `L0_vis_place_rgbdt_700demos_imagenet` (0.536) on the same episodes (sign
   test, p < 0.05), and per cell against that cell's best baseline (item 5 of the results of step 2). One seed is a
   first reading.
2. The pick and the place apart: the share of episodes with the glass picked, and placed among the picked, next to
   the rgbdt baseline (0.605, 0.886).
3. Which part carries it, on the same episodes, p < 0.05: `hcg2_full` against `hcg2_cue` (gating the features),
   `hcg2_cue` against `hcg_cue` (the place cue), `hcg2_cue` against `hcg2_blind` (the images).
4. The cues inside the episodes: the share with the grasp cue within 1.5 cm of the glass and with the place cue
   within 1.5 cm of the coaster's centre, and the success with each right and wrong.
5. Next to them, not a test: success in the cells in which the colour camera shows the glass (0, 2, 3, 6, 7, 10) and
   in those in which it does not (1, 4, 5, 8, 9), for `hcg2_full` and `hcg2_cue`.

Not in this step: seeds 43 and 44, the controls of round 2 of the pick level (one network, gate on map numbers, no
gate), the references. Results are reported as they come out, whatever they are.

## Files

- `policy/readout_place.py` (the fixed read-out), `policy/place_pick_compare.py` (the first close on both levels).
- `sim/configs/L0_vis_place.yaml`, `sim/assets/gen3_lite/place_scene.xml`, `sim/config.py` (`task`, `place_radius`),
  `sim/env.py` (`_place_target`, `placed`), `sim/teacher.py` (carry and place), `policy/glass_gym_env.py` (labels).
- `sim/place_check.py` (teacher check without rendering the steps), `sim/figure_place.py` (the figure).
