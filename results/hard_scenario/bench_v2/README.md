# Visibility benchmark, second version: design draft for review

**Status (11/26): a draft. The states of section 3 and the level of task T1 are implemented (section 10); nothing is
collected, trained or frozen.** Decision of the user 11/26:
a second version of the benchmark is designed in one piece, tried with a pilot, frozen once, and then everything is
run on it once. The levels that exist (`L0_vis`, `L0_vis_ell`, `L0_vis_place`) and their results are not touched: they
stay as the development record and as the fallback for the paper if this version is late. Points marked **[decide]**
are for the user.

## 1. Why a second version

Three things about the first version that a reader of the paper can object to, and that cannot be mended by adding
to it:

1. *The sensor failures are numbers we picked* (40% holes, 3 degC, alpha 0.2). Section 2 measures what real sensors do.
2. *The demos and the evaluation use different contact models* (`L0_vis` demos, `L0_vis_ell` evaluation).
3. *The one task is solved by one point.* A policy that gets the cue and the joints and no image is at 0.888.

What stays: three sensors (colour, depth, thermal) and the joints; three states per sensor (shows / faint / blind) and
the cells built from them; the robot, the camera, the image size, the action space; 700 demos of a scripted teacher;
test sets of 660 episodes with every (cell, object) pair equally often; three seeds, the same episodes for every run,
sign tests; the comparison rules of Stage 6.

## 2. What real sensors do with real glass (measured 11/26)

`python real_data/glass_visibility.py --n 800` on the five public datasets of `real_data/` (per connected glass region
of the mask; frames drawn evenly from every scene or sequence), and `python sim/measure_visibility.py L0_vis -n 440`
with the same functions on first frames of the simulator at the sensors' own resolution. Outputs in
[`measurements/`](measurements/).

**Depth: the share of the glass's pixels without a reading, and with a reading from behind the glass.**

| source | sensor | regions | missing: p10 / median / p90 | reads more than 2 cm behind: median | |
|---|---|---|---|---|---|
| MJU-Glass | Orbbec Gemini Pro, structured light | 2989 | 0.78 / 1.00 / 1.00 | no ground truth | the glass is a hole |
| T-ROBI | Ensenso, active stereo | 1960 | 0.88 / 0.94 / 0.96 | 0.00 | the glass is a hole |
| TRansPose | RealSense L515, lidar | 1563 | 0.00 / 0.00 / 0.34 | 0.91 | the sensor sees through the glass (0.92 of the regions) |
| `L0_vis`, depth shows | | 204 | 0.34 / 0.38 / 0.44 | 0.54 | between the two kinds |
| `L0_vis`, depth faint | | 58 | 0.18 / 0.20 / 0.26 | 0.70 | the upper tail of the lidar |
| `L0_vis`, depth blind | | 178 | 0.00 / 0.00 / 0.00 | 0.90 | the lidar's median |

Real depth on glass comes in two kinds, by the sensor: a hole over nearly the whole glass, or the surface behind it.
"Blind" and "faint" of `L0_vis` are a see-through sensor as measured. "Shows" (40% holes) is not what either kind does.

**Thermal: the contrast between the glass and what is around it.**

| source | what | regions | p10 / p25 / median / p75 / p90 | weak |
|---|---|---|---|---|
| TRansPose | tabletop glass objects at rest, in units of the camera's noise | 1563 | 10 / 26 / 47 / 67 / 103 | below 3: 0.032; below 10: 0.096 |
| the same, with the gain of the thermal model (42.67 counts per degC) | degC | | 0.21 / 0.54 / 0.98 / 1.41 / 2.16 | |
| RGB-T Glass | panes (windows, doors), degC | 2290 | 0.01 / 0.05 / 0.14 / 0.37 / 0.88 | below 0.1 degC: 0.41 |
| `L0_vis`, thermal shows | noise units (degC) | 217 | median 488 (10.2 degC), p10 191 (4.0 degC) | |
| `L0_vis`, thermal faint | | 63 | median 82 (1.7 degC) | |
| `L0_vis`, thermal blind | | 160 | median 1.4 (0.03 degC), p90 3.1 | |

A glass object standing on a table differs from its surroundings by about 1 degC in the thermal image (the thermal
model of the simulator was fitted to the same camera). "Shows" of `L0_vis` is ten times that: it is a glass with
something warm or cold in it. "Faint" of `L0_vis` (1.7 degC) is a strong case of a glass at rest. "Blind" of `L0_vis`
exists (3% of the tabletop regions are below 3 noise units, 41% of the panes below 0.1 degC).

**Colour: |median luminance of the glass - median of a ring around it|, grey levels of 255.**

| source | regions | p10 / p25 / median / p75 / p90 | below 3 |
|---|---|---|---|
| TRansPose, tabletop objects | 1563 | 2.7 / 7.6 / 19.9 / 43.6 / 76.7 | 0.11 |
| T-ROBI, objects in a bin (grey camera) | 1960 | 3 / 7 / 14 / 19 / 26 | 0.09 |
| panes (MJU-Glass, RGB-T Glass, RGBD-GSD) | 8445 | medians 54 to 82 | 0.02 to 0.04 |
| `L0_vis`, rgb shows (alpha 0.5) | 202 | 9.0 / - / 26.7 / - / 44.4 | |
| `L0_vis`, rgb faint (alpha 0.2) | 61 | 4.0 / - / 12.8 / - / 23.2 | |
| `L0_vis`, rgb blind (alpha 0.001) | 177 | 0.0 / - / 0.0 / - / 0.3 | |

The colour states of `L0_vis` lie where real tabletop glass lies: "shows" at its median to upper quartile, "faint" at
its lower quartile, "blind" below its lowest tenth.

**The three sensors on the same real objects** (TRansPose, 1548 regions): depth sees through 0.93 of them, thermal is
strong (10 noise units or more) on 0.90, colour contrast is 10 grey levels or more on 0.68; all three weak at once:
0.002. In 0.85 of the regions the depth camera sees through the glass while the thermal camera shows it clearly.
This is the premise of the benchmark, measured.

**What these measurements are not.** One thermal camera, one lidar, two hole-type depth sensors; the objects, scenes
and distances are those of the datasets, not ours. They fix the ranges of the states, not how often each cell occurs
on a given robot. The colour measure is a contrast: the renderer does not refract or reflect, so glass in the
simulator does not look like glass even where its contrast is right. That limit stays and is stated in the paper.

## 3. The visibility states of the second version

Each state is a cause that exists, with its range from section 2.

| sensor | state | cause | second version | first version |
|---|---|---|---|---|
| depth | shows | a hole-type sensor (structured light, stereo) | 0.80 to 1.00 of the glass's pixels missing, drawn per episode | 0.40 |
| depth | faint | a see-through sensor with some dropout | 0.10 to 0.35 missing | 0.20 |
| depth | blind | a see-through sensor (lidar, time of flight) | none missing, the surface behind | the same |
| thermal | shows | a glass at rest, usual contrast: the middle 80% of the tabletop regions | 10 to 103 noise units (0.2 to 2.2 degC), either sign | 4 to 18 degC |
| thermal | faint | a glass at rest, low contrast | 3 to 10 noise units (0.06 to 0.21 degC) | about 1.7 degC |
| thermal | blind | glass and background in equilibrium | below 3 noise units | below about 5 |
| colour | shows | usual contrast against the table | alpha 0.5 (median contrast 27) | the same |
| colour | faint | low contrast | alpha 0.2 (median 13) | the same |
| colour | blind | the glass is not in the colour image (corrected 11/26, section 10) | alpha 0.001 | the same |

**What the colour states are and are not (11/26, after section 12).** They set how much of the glass's body shows
in the colour image, at the contrast measured on real glass. They do not model how clear glass looks: the renderer
blends a tinted solid over the scene and does not refract or reflect, and a network that learned real glass does not
take the simulator's glass for glass (sections 11 and 12). "Colour shows" is therefore closest to a transparent
object whose body is visible in colour (frosted or tinted glass, a plastic cup), not to clear glass with its
highlights. This is stated in the paper as a limit of the benchmark; clear glass in colour is left to the real robot.

**What this does to the difficulty, expected and not known.** Where depth shows the glass it gets easier (the glass
is one large hole). Where only thermal shows it, it gets harder: "shows" of this version is about "faint" of the
first, the cell in which the method is lowest today (0.71 on `L0_vis_ell`, baselines 0.17), and the new "faint" is two
to three grey levels in the 8-bit thermal image, close to its noise. The pilot (section 8) measures it before anything
is frozen.

**[decide] Glass with warm or cold content** (today's thermal "shows", 5 to 35 degC). Proposed: not among the cells;
it can be added later as its own test set ("filled glass"), since the simulator has liquids. The alternative keeps it
as a fourth thermal state and makes the table of cells larger.

**Cells.** The eleven cells and their weights as in `L0_vis` (seven combinations of shows / blind, three with one
sensor faint and the others blind, one with all three faint), with the states above.

## 4. Tasks

Picking alone is no longer a task of the benchmark (decision of the user). Three tasks, each adding one demand:

| task | what | asks for | episode | objects |
|---|---|---|---|---|
| T1 `place` | pick the glass and put it down upright on a coaster | where | up to 240 steps | the five glass shapes |
| T2 `align` | pick an elongated glass that lies at a random angle, across its narrow side, and put it down on a rectangular mat in the mat's direction | where and in which direction | up to 240 steps | new: elongated shapes |
| T3 `two` | two glasses, two coasters: put one on each | two targets; the sensors that show the first glass need not show the second | up to 480 steps | two of the five shapes at once |

- **T1** is `L0_vis_place` with the states of section 3 and one contact model for demos and evaluation (elliptic
  cones, as `L0_vis_place` has). Whether the glass was picked stays in the record of every episode, so the pick is
  still read.
- **T2**, new. Glass boxes whose long side is wider than the open gripper, so that only a grasp across the narrow
  side holds; the angle on the table is random and the mat has its own. Success: the glass upright on the mat, its
  centre within 2 cm and its direction within 15 degrees of the mat's, released. To build: the shapes (three for
  training, proposed), the teacher's wrist angle at the grasp and at the place, the mat, the success test. The method's
  cue becomes a grasp pose (point and angle), read in hindsight from the gripper at its first close as the point is now.
- **T3**, new. Each glass has its own visibility states: its own alpha, its own depth dropout, its own temperature.
  In half of the episodes the two are shown by different sensors, so the sensor to trust changes within the episode.
  Success: both glasses upright on a coaster each, released. To build: two glasses on the table at once (today one
  is on the table and the others are parked), the depth dropout per object, the teacher doing the pick-and-place
  twice, the success test, pairs of cells in place of cells. The pairs are fixed after the first renders
  **[decide after the pilot]**; proposed start: both glasses in the same cell (four pairs), in different cells (six
  pairs), with a faint one (two pairs).

**Cost.** A T1 run is about 8 hours (measured on `L0_vis_place`), T2 about the same, T3 about twice that with 700
demos. **[decide after the pilot]** whether T3 keeps 700 demos.

## 5. Test sets

| set | what | training needed |
|---|---|---|
| `j` in distribution | the scenes of the training set, new seeds | no |
| `k` held-out scenes | backgrounds, lighting and distractors not in the training set | no |
| `n` unseen combinations | the 15 combinations of states that are not among the eleven cells | no |
| `u` unseen objects | two glass shapes that are not in the training set (to build) | no |
| `v` validation | the training scenes, seeds apart from `j`; used for every choice of a setting, never reported | no |
| left-out cells | trained without cells 4 to 9, tested on them (`nosingle`) | one training set more, T1 only |

The headline numbers are `j` and `k` (660 episodes each, macro average over the cells, and the lowest cell). `n`, `u`
and the left-out cells are the generalisation table.

## 6. Policies in the table and how they are set

- Categories as now: (a) sensor images and joints only (the plain baselines, the baselines from the literature, the
  method); (b) extra supervision (simulator masks); (c) references that read the simulator (the true grasp point).
- **[decide] A stated tuning budget.** Proposed: on T1, seed 42, scored on `v`: each of plain rgbdt, the method's
  arm `cue` and the one-network control is trained with the four settings {101, 201 epochs} x {learning rate 1e-4,
  3e-4}; the best of each on `v` is its setting for every task and seed. Twelve training runs, about a day. Without
  it every policy keeps the benchmark's one setting, as now.
- On all three tasks, three seeds: the four plain baselines, `cue`, the one-network control. On T1 only: the three
  baselines from the literature, the second policy (ACT head) without and with the cue, the true grasp point.

## 7. What is released with it

The demos of every task, the last checkpoint of every run in the tables, the level files and test-set manifest, one
command per run and per evaluation, the per-episode records behind every number, and the two measurement scripts of
section 2 so that the states can be checked against other sensors.

## 8. Steps, each with a stop

| step | what | until (about) | stop if |
|---|---|---|---|
| 1 | this draft reviewed, the points marked [decide] settled | 12/3 | |
| 2 | the states of section 3 and T1 implemented; `sim/measure_visibility.py` on the new level lands in the ranges of section 2; teacher success 0.95 or more in every cell | 12/8 | the ranges cannot be met |
| 3 | **pilot** on T1, seed 42: plain rgbdt, plain rgbd, `cue`, the one network; a single-sensor reference per cell | 12/11 | a cell that no reference solves (below 0.30), or everything above 0.95 |
| 4 | T2 and T3 implemented, teacher success checked, a plain rgbdt run of each | 12/17 | a task the plain baseline cannot start on (below 0.05) and a reference cannot solve |
| 5 | **freeze**: the level files, the test sets, the run list and the read-outs committed | 12/17 | |
| 6 | everything once: about 870 GPU hours, nine to ten days on four GPUs | 12/27 | |
| 7 | read-outs, figures, text | January | |

If step 3 or 4 stops, the states or the task are changed and the step repeated; nothing has been frozen by then.
If the freeze slips past the end of December, the paper for the January deadline is written on the first version,
whose runs are finishing now.

## 9. For the user to decide now

1. The states of section 3, in particular thermal "shows" at about 1 degC in place of about 10.
2. Glass with warm or cold content: outside the cells (proposed) or a fourth thermal state.
3. The tuning budget of section 6: with (proposed) or without.
4. T3 with two coasters and free order (proposed), or with a fixed order.

## 10. Progress

**The points of section 9 (11/26).** The user leaves them to the proposals. So: (1) the thermal states are not
chosen now: the pilot runs T1 with the states of section 3 (`V2_place`) and with the thermal states of the first
version (`V2_place_warm`: 5 to 35 degC, about 3 degC, 0.1 degC) side by side, and the user chooses with the two
tables; (2) glass with warm or cold content is outside the cells; (3) the tuning budget is kept; (4) T3 has two
coasters and a free order. All four can still be changed until the freeze. Point (1) is replaced by the rule of 11/26 below
("The thermal states are fixed by a rule").

**Step 2, the states and the level of T1 (11/26).** New settings in `sim/config.py` and `sim/env.py`: `show_holes`
(depth shows, a range), ranges for `faint_holes` and `clear_alpha`, `thermal.show_contrast` (thermal shows, a contrast
against the table in place of a gap from the room). Each has an RNG stream of its own and is off unless a level sets
it: `sim/regression_test.py` gives 190 of 190 arrays identical for `L0_vis`, `L0_vis_place`, `L0_vis_ell` and the
default scene before and after. Levels `sim/configs/V2_place.yaml` and `V2_place_warm.yaml` (drafts, ids 38 and 39).
`python sim/measure_visibility.py V2_place -n 440` ([`measurements/sim_V2_place_visibility.txt`](measurements/sim_V2_place_visibility.txt)),
next to the real ranges of section 2:

| sensor, state | `V2_place`: p10 / median / p90 | real |
|---|---|---|
| depth shows, share missing | 0.80 / 0.88 / 0.97 | 0.78 / 1.00 / 1.00 and 0.88 / 0.94 / 0.96 (the two hole-type sensors) |
| depth faint, share missing | 0.12 / 0.25 / 0.33 | the lidar's upper quarter: 0.07 (p75) to 0.34 (p90) |
| depth blind, share missing; reading from behind | 0.00; 0.90 | the lidar: 0.00; 0.91 |
| thermal shows, noise units | 19 / 61 / 93 | 10 / 47 / 103 |
| thermal faint, noise units | 2.8 / 6.6 / 9.8 | 3 to 10: 6% of the tabletop regions |
| thermal blind, noise units | 0.4 / 1.5 / 3.4 | below 3: 3% |
| colour shows, grey levels | 9 / 27 / 44 | 2.7 / 20 / 77 |
| colour faint | 4 / 13 / 23 | the lower quarter: 7.6 and below |
| colour blind (after the correction below) | 0.0 / 0.0 / 0.3 | below 3: 11% |

The thermal values of the level file are those of `sensors.matching_temp`; the image shows about 0.6 of them
(the blur of the 160x120 thermal image on a glass of about 50 pixels), which is why they are larger than the
contrasts they give. "Shows" is drawn evenly over the range, so its median is above the real one (61 against 47).

**Step 3, the pilot (listed 11/26 07:12 KST, before any demo of the two levels is collected, any cue fitted or any
policy trained).** Task T1 on the two draft levels, seed 42, the protocol of `L0_vis_place` (700 demos of the scripted
teacher, 101 epochs, last checkpoint, ImageNet encoders, episodes of at most 240 steps, 660 episodes on each of
`j_cells11` and `k_cells11_heldout`).

| run, for `<level>` = `V2_place` and `V2_place_warm` | what |
|---|---|
| `<level>_rgbdt_700demos_imagenet` | the plain rgbdt baseline |
| `<level>_rgbd_700demos_imagenet` | the plain rgbd baseline |
| `<level>_rgbdt_700demos_hcg2_cue` | the arm `cue` with the grasp cue and the place cue of the method, fitted on the level's demos |
| `<level>_rgbdt_700demos_hcg2_cue_early` | the same with the grasp cue of one network on all sensors (the place cue is the method's: `hindsight_cue.py --event release` exists for the default fit only) |

Before them, per level: `python sim/collect_demos.py --n 700 --workers 4 --config sim/configs/<level>.yaml --out
results/data/sim_demos_<level>_700.zarr`, then `python policy/hindsight_cue.py <demos> --level <level> --out
data/hcue/<level>_700` (the grasp cue), `... --mode early --out data/hcue/<level>_700_early` and `... --event release
--out data/hcue/<level>_700_release`. Commands of the runs: those of steps 2 and 3 of
[`../place_bench/README.md`](../place_bench/README.md) with `GROUP=bench_v2`, the level, and for the two runs with cues
`CUES=2 TRAIN="policy/train_cuegate.sh cue"` and `task.hcue_dir=data/hcue/<level>_700[_early]
task.hcue_dir2=data/hcue/<level>_700_release`. Eight runs through the queue of the fusion-vis worktree, after
everything queued before them; a run that fails is reported and not started again.

Read-out, fixed now (`python policy/readout_v2_pilot.py > readout_pilot.txt`): per level the success per cell of the
four runs and the share of episodes with the glass picked; the two stops of step 3 (a cell in which every run is
below 0.30; every cell of every run above 0.95); `cue` against plain rgbdt and against the one network on the same
episodes (sign test); the two levels side by side. The pilot is one seed and decides two things only: whether the
states are in a range that can be measured, and which thermal states the user takes. Nothing of it enters the tables
of the frozen benchmark.

**The thermal states are fixed by a rule (11/26 11:21 KST; decision of the user; none of the eight pilot runs has
started, no policy has been trained on either level).** Choosing between `V2_place` and `V2_place_warm` with the two
tables of the pilot in hand would leave a choice open that the method's numbers could steer. It is closed now:

1. The thermal states of the benchmark are those of section 3, the ranges measured on real glass at rest
   (`V2_place`). `V2_place_warm` is not a candidate for the benchmark's level. Its four pilot runs stay and are
   reported as a sensitivity analysis: what the same runs give when the thermal contrast is about ten times the real
   one.
2. The only ground for changing a thermal state is that a cell cannot be solved, and the evidence is a reference
   policy, not the method. If on `V2_place` a cell is below 0.30 in all four pilot runs, one run is added:
   `V2_place_rgbt_430demos_thermal_imagenet`, the plain rgbt baseline trained on the 430 demos of the cells in which
   thermal shows the glass fully or faintly (cells 0, 1, 2, 4, 9, 10; `policy/subset_demos.py`), seed 42, the protocol
   of the pilot. It uses the cells to choose its demos, so it is a reference and not a baseline (as stage 1 of
   [`../grid_bench/README.md`](../grid_bench/README.md)). A cell counts as not solvable if this reference is below
   0.30 in it as well. A cell that a run or the reference solves stays as it is, however low the other runs are.
3. What is changed then is fixed too, and stays inside the measured range. If the cells that are not solvable are
   cells in which thermal is faint (9, 10): thermal "faint" becomes the real p10 to p25 (10 to 26 noise units) and
   "shows" the real p25 to p90 (26 to 103), the level file is fitted to these with `sim/measure_visibility.py`, and
   the pilot runs of `V2_place` are repeated once. If a cell in which thermal shows the glass fully is not solvable,
   or a cell whose thermal state is blind, the stop of section 8 applies as written: the states are changed with the
   user, and `V2_place_warm` is not the replacement by itself.

The read-out script of the pilot is unchanged; its side-by-side table of the two levels is the sensitivity analysis
of point 1. The user can still overrule any of this until the freeze.

**Step 4, task T2 `align` in the simulator (11/26 08:39 KST; no demo collected, no policy).** `task: align`
(`sim/config.py`), scene `sim/assets/gen3_lite/align_scene.xml`, level `sim/configs/V2_align.yaml` (a draft, id 40; the
states of `V2_place`).

- *Glasses*: boxes with the long side wider than the open gripper: `slab` 14 x 4 x 8 cm, `bar` 15 x 3.5 x 7 cm, `block`
  13 x 4.5 x 10 cm for training, `tile` 12 x 4 x 7 cm for a test set of unseen shapes. One lies on the table at an
  angle drawn from 0 to 180 degrees.
- *Mat*: 17 x 7 cm, dark with a light stripe along it, at a free spot within reach (as the coaster of T1) and at an
  angle of its own.
- *Success*: the glass upright on the mat, its centre within 2 cm of the mat's and its long side within 15 degrees of
  the mat's (either way round), at rest, the gripper away. Failure labels as T1, plus `place_turned`. The record of
  an episode has `yaw_error`.
- *Teacher*: the plan of T1 with the tool turned about the vertical: to the glass's angle on the way there, to the
  mat's angle while the glass is carried (at most 6 degrees per step). The last arm joint has room in one direction
  only, so the angle is taken from the half turn -170 to 10 degrees. 60 of 60 scenes (seeds 0 to 59, 20 per shape);
  121 steps in the median, 142 at most, of the 240 allowed.
- The test sets deal the three training shapes of this task; the object ids of its records continue after those of
  the five glasses (5 to 8).
- Every existing level is as it was: `sim/regression_test.py`, 190 of 190 arrays identical for `L0_vis`,
  `L0_vis_place`, `L0_vis_ell` and the default scene.

Not done for T2: the cue of the method with the grasp angle (the gripper's angle at its first close, read in
hindsight as the point is), a plain rgbdt run, the solvability check per cell. T3 has not been started.

**Unseen objects (11/26; the test set `u` of section 5).** `objects: unseen` (`sim/config.py`) and the test set
`u_cells11_unseen` (the eleven cells, the training scenes, glass shapes that are in no training set). For T1 two
shapes in `sim/assets/gen3_lite/place_unseen_scene.xml`: `tallglass` (a straight glass, 5.6 cm wide and 13 cm tall) and
`carafe` (a 7 cm wide body with a 3.6 cm wide neck and a lip); for T2 the `tile`. The scripted teacher on them through
the evaluation env: 22 of 22 for each of the two shapes of T1, 44 of 44 for the tile. Object ids of the records: 8
(tile), 9 and 10. Existing levels unchanged (regression test).

**Step 4, task T3 `two` in the simulator (11/26 09:44 KST; no demo collected, no policy).** `task: two`, scene
`sim/assets/gen3_lite/two_scene.xml` (a second coaster), level `sim/configs/V2_two.yaml` (a draft, id 41), test sets
`j_pairs12` and `k_pairs12_heldout`.

- *Two glasses* of different shapes from the five; the second stands at least 7 cm (edge to edge) from the first. At
  5 cm the open gripper, which is 14 cm wide, knocked the neighbour over in 2 of 120 teacher episodes.
- *A cell is a pair*: the colour, depth and thermal state of the first glass and of the second. Each glass has its
  own alpha, its own depth dropout or see-through, its own temperature. Twelve pairs to start with: both shown by all
  three; both by depth alone, by thermal alone, by colour alone; six pairs in which the two are shown by different
  sensors; both faintly; one fully and one faintly. Weights 20, 3 x 8, 6 x 12, 2 x 12 of 140.
- *Two coasters*, each clear of both glasses, of the distractors and of the other coaster. The scene is made so that
  it can be done in an order that is visible in it: the glass nearer to the gripper first, to the coaster nearer to
  that glass if the straight way there passes over nothing and else to the other coaster; then the second glass to
  the coaster that is left. That is the teacher's plan; a policy may use any order.
- *Room on the table*: 0 to 2 distractors (1 to 3 in the other tasks), and where even then no such plan exists the
  distractors are taken off the table. Of 600 scenes 592 have a plan as drawn and 8 after the distractors are
  removed; none is left without one. With 1 to 3 distractors one scene in ten had no free spot for the coasters.
- *Success*: each glass upright on a coaster of its own, at rest, let go (the tool above it or 8 cm away in the
  plane, since the arm goes on to the other glass). The record of an episode has `n_placed`, `picked`, `picked2`; an
  episode that fails is labelled `placed_0` or `placed_1`.
- *Teacher*: the pick-and-place of T1 twice. `python sim/task_check.py V2_two -n 240`: 240 of 240, every pair and
  every shape; 226 steps in the median, 256 at most, of the 480 allowed. With the final distance rule (11/26 10:57
  KST): 600 training scenes all have a plan (583 as drawn, 17 after the distractors are removed); the teacher
  succeeds in 240 of 240 training scenes and in 180 of 180 episodes of each test set, `j_pairs12` and
  `k_pairs12_heldout` (every pair 15 of 15, every shape 36 of 36).
- Existing levels unchanged (`sim/regression_test.py`, 190 of 190 arrays identical for the four configs).

Not done for T3: the cue of the method for two targets (four cues: which glass first, then the second), a plain rgbdt
run, the solvability check per pair. The pairs and the number of demos are fixed after the first runs.

**Checking a task** (`python sim/task_check.py LEVEL [-n N] [--test_set SET]`): the teacher on consecutive seeds of the
training scenes or of a test set; success per shape and per cell, steps, every failure.

**Correction: colour blind (11/26 10:27 KST).** The draft set "colour blind" to an alpha of 0 to 0.03, the contrast of
the weakest tenth of real tabletop glass (below 3 grey levels). In the simulator's colour image, which has no noise,
that is not blind: against the same scene without the glass about 30 pixels of the 96 x 96 image differ, by 4 grey
levels in the median and up to 15, and only one episode in ten is unchanged. Networks find that: the colour network of
the cue has its peak on the glass in 0.37 of the colour-blind frames of `V2_place` (0.15 on `L0_vis_place`, which is
chance) and its gate is at 0.54 there, and a network reading the glass's direction from the colour patch is within 15
degrees in four of five colour-blind episodes. It was the test of the angle networks below that showed it, before any
policy was trained. So colour blind is alpha 0.001 again, as in the first version: no pixel differs. The demos of
`V2_place`, `V2_place_warm` and `V2_align` collected before the correction and the cues fitted on them are discarded
and made again (`data/hcue/v2_prepare.log`); the eight pilot runs had not started.

**The cue of the method for T2: the hindsight angle (11/26; `policy/hindsight_angle.py`, in work).** A demo of T2 says
in hindsight how the tool was turned when the gripper closed and when it opened again; these angles are the glass's
and the mat's direction within 2 degrees (700 demos, against the simulator, checked once). Per sensor a small network
reads the direction in the image from the patch around the cue's point, and the sensors are added with the cue's gate
entries. What the first fits showed (700 demos of `V2_align` collected before the correction above; out of fold; the
error of the tool's angle; the cells of the simulator stood in for the gates, since the cue was not fitted yet):

| patch, training | colour, where it shows the glass | depth | thermal |
|---|---|---|---|
| 33 x 33, not turned | 34 degrees in the median (1 degree on its own training patches) | 2.5 | 19 |
| 33 x 33, every patch turned by a random angle with its label | 46 (chance) | 3.4 | 44 |
| 17 x 17, turned | 1.3, within 15 degrees in 1.00 | 4.1, in 0.89 | 20.9, in 0.39 |
| 17 x 17, turned, squared error on a vector that is not normalized | 1.6 to 1.7, in 1.00 | 2.2 to 2.6, in 1.00 | 4.7 to 6.3, in 0.83 to 0.90 |

In the large patch the mat, the distractors and the arm are what a network sees; in the small one the glass. With
the squared error the length of a network's vector says how sure it is, and adding the sensors' vectors lets a sure
sensor outweigh an unsure one: the estimate is within 15 degrees in 0.96 of the demos (0.86 to 1.00 per cell). The
last row was fitted with every sensor trusted everywhere and before the correction, so its colour network also read
the trace of the "blind" glass; the numbers with the real cue and the corrected level follow
(`data/hang/V2_align_700/report.txt`). The policy gets the two angles as a further observation (`hang`: cos and sin
of twice the angle at the grasp and at the place; `policy/config/task/glass_pick_hcue2a.yaml`, `ANGLE=1` of
`policy/train_cuegate.sh`).

**Step 4, the runs on T2 and T3 (listed 11/26 14:02 KST, before any of them is started; the demos of T3 are not collected
yet).** Whether a policy can start on the two tasks. Seed 42, the protocol of the pilot (101 epochs, the last
checkpoint, ImageNet encoders), through the queue after the pilot's lines; a run that fails is reported and not
started again.

| run | what | evaluation |
|---|---|---|
| `V2_align_rgbdt_700demos_imagenet` | plain rgbdt on T2 | `j_cells11`, `k_cells11_heldout`, 660 episodes each, at most 240 steps |
| `V2_align_rgbdt_700demos_hcg2a_cue` | the arm `cue` of the method on T2 with the grasp cue, the place cue and the hindsight angle of each (`ANGLE=1 CUES=2 TRAIN="policy/train_cuegate.sh cue"`, the fits of `data/hcue/V2_align_700[_release]` and `data/hang/V2_align_700[_release]`) | the same |
| `V2_two_rgbdt_700demos_imagenet` | plain rgbdt on T3 | `j_pairs12`, `k_pairs12_heldout`, 360 episodes each (30 per pair), at most 480 steps |

Commands: `GROUP=bench_v2 TOPK=0 NTEST=66 NEVAL=660 TEST_SETS="j_cells11 k_cells11_heldout" TAG=imagenet
EXTRA="training.seed=42 task.env_runner.test_set=j_cells11 task.env_runner.max_steps=240
policy.obs_encoder.rgb_model.weights=IMAGENET1K_V1" bash scripts/run_experiment.sh V2_align rgbdt 700`; for T3 the same
with `NTEST=60 NEVAL=360`, the test sets `j_pairs12 k_pairs12_heldout`, `max_steps=480` and `V2_two`; the method's run
on T2 as the pilot's `hcg2_cue` line with `ANGLE=1`, `TAG=hcg2a_cue` and the four folders above. The demos of T2 are
the 700 of 11/26 (after the correction of colour blind); those of T3 are collected by the run. The method's run is
put into the queue once its fits exist and a start test of a few steps has passed; the two plain runs passed such a
test before they were queued. The method is not run on T3: its cue for two targets is not written.

Read-out, fixed now (`python policy/readout_v2_tasks.py > readout_tasks.txt`): per task the success per cell (T3: per
pair) of each run, the share of episodes with the glass picked (T3: with one and with both glasses on a coaster),
the outcome labels of the failures, for T2 the angle between glass and mat at the end. The stop of step 4 as in
section 8: a task on which plain rgbdt is below 0.05 and no run reaches 0.30 in any cell is changed with the user
and its runs are repeated. These runs decide that and, with the pilot, the numbers of demos of T2 and T3 and the
pairs of T3; nothing of them enters the tables of the frozen benchmark.

**Order in the queue (11/26 14:12 KST; the user wants the benchmark confirmed first).** The runs that decide the freeze
go before the remaining runs of the first version: the two plain runs of `V2_place` and the plain runs of T2 and T3
now, and the method's runs on `V2_place` (`hcg2_cue`, `hcg2_cue_early`) and on T2 as soon as their fits exist. The
four runs of `V2_place_warm` are a sensitivity analysis and stay at the end. The order changes when a run starts and
nothing else; the listings and read-outs above are as they were.

*11/26 17:51 KST:* the plain run of T3 is moved before that of T2 (neither has started). Its demos and its episodes are
twice as long, so it is the longest of the runs that decide the freeze and should not be the last to end.

*11/26 19:36 KST: the method's run on T2 is in the queue*, after the plain run of T2 (`V2_align_rgbdt_700demos_hcg2a_cue`,
the command as listed). Its fits exist since 19:25 KST (700 demos, out of fold, the rule `consensus`):

- grasp cue: within 1.5 cm of the hindsight grasp point in 0.81 of the demos. At reset on the 660 scenes of
  `j_cells11` it is within 1.5 cm of the glass in 0.79: cells 0 to 3 and 10 0.92 to 1.00, cells 5 to 7 0.83 to 0.85,
  cell 8 (depth faint alone) 0.67, cell 4 (thermal alone) 0.58, cell 9 (thermal faint alone) 0.08;
- place cue: within 1.5 cm in 0.89 of the demos, 0.82 to 0.94 by cell;
- hindsight angle of the grasp: within 15 degrees in 0.94 of the demos (cell 9 0.61, cell 8 0.85, the others 0.90 to
  1.00); of the place: 1.00.

In cell 9 the cue does not find the glass (thermal alone, in its faint range of 3 to 10 noise units). Whether that
cell can be solved at all is what the rule of the thermal states above decides, on `V2_place` and by the pilot;
nothing is decided from these fits. A start test in the debug mode of Diffusion Policy with the four folders (six
training steps, two rollouts of 11 episodes, the final evaluation on 11 episodes of each test set; the episode
records carry the cue, the angles, `picked` and `yaw_error`) ran through. Its files are deleted.

**A cache that was out of date (11/26 19:42 KST).** The fits of `V2_place` that the preparation started at 19:25 ended
within three minutes. `policy/hindsight_cue.py` keeps the maps of its networks in `maps.pt` and reads them again when
the path of the demo set and the settings are the same, and the maps in `data/hcue/V2_place_700`, `..._release` and
those of `V2_place_warm` were of this morning (08:00 to 10:04 KST): made from the demos that were collected before
the correction of colour blind and collected again at 10:27 and 10:36. Those cues came from networks trained on the
earlier images; their report showed the 0.37 of the correction again, which is how it was seen. The folders are
removed and fitted again from the current demos (from 19:42 KST). Not affected: `V2_align` (maps of 18:24 and 19:01
KST, demos of 10:45) and `V2_place_700_early` (the one network keeps no maps; fitted 19:28 to 19:40). No policy run
has used a cue of `V2_place`: the method's pilot lines have not started. The cache does not look at the demo set
itself. The code is not changed while runs execute from this worktree; until then a cue folder is removed whenever
its demo set is collected again, and the preparation checks the dates.

*11/26 21:01 KST: the cues of `V2_place` are fitted again* (maps of 20:17 and 20:54 KST, demos of 10:36; the
preparation wrote "V2_place ready" at 20:59). From the current demos, the rule `consensus`:

- grasp cue: within 1.5 cm of the hindsight grasp point in 0.82 of the 700 demos. At reset it is within 1.5 cm of the
  glass in 0.81 of the 660 scenes of `j_cells11` and 0.78 of `k_cells11_heldout`: cells 0 to 3, 5 to 7 and 10 0.82 to
  1.00, cell 8 (depth faint alone) 0.65 and 0.48, cell 4 (thermal alone) 0.60 and 0.60, cell 9 (thermal faint alone)
  0.05 and 0.03;
- place cue: within 1.5 cm in 0.97 of the demos;
- the one network on the five channels (`V2_place_700_early`, fitted 19:28 to 19:40): 0.80 of the demos; at reset
  0.77 and 0.74, cell 4 0.27 and 0.15, cell 9 0.07 and 0.03, cell 8 0.55 and 0.57;
- own peak of a sensor within 1.5 px where it shows the glass / faintly / not: colour 0.98 / 0.92 / 0.15, depth 1.00 /
  0.89 / 0.16, thermal 0.70 / 0.21 / 0.12.

As on T2, neither cue finds the glass in cell 9; the rule of the thermal states decides on that cell from the four
pilot runs, not from these fits. With the cues there, the two method runs of the pilot on `V2_place` are moved forward
among the lines that have not started (the commands as listed, no line added or removed): `V2_two` plain,
`V2_place_rgbdt_700demos_hcg2_cue`, `V2_align` plain, `V2_align_rgbdt_700demos_hcg2a_cue`,
`V2_place_rgbdt_700demos_hcg2_cue_early`, then the lines of the first version; the four runs on `V2_place_warm`
(sensitivity only) stay at the end.

*11/26 22:29 KST: the cues of `V2_place_warm` are fitted again too* (22:29 KST, all files later than its demos of
10:45; the preparation is at its end). The same numbers for the warm variant, which only the sensitivity runs at the
end of the queue use: the grasp cue at reset is within 1.5 cm of the glass in 0.89 of `j_cells11` and 0.84 of
`k_cells11_heldout` (cell 9 0.68 and 0.60, cell 4 0.87 and 0.72, cell 8 0.63 and 0.57), the one network in 0.91 and
0.89 (cell 9 0.75 and 0.68, cell 4 1.00 and 0.92, cell 8 0.68 and 0.73); the own peak of the thermal sensor is within
1.5 px where it shows the glass / faintly / not in 0.93 / 0.75 / 0.07. With the warmer glass the cue finds it in cell
9, and the one network is not behind the cue of the method in the cells with thermal alone. Fits only; what the
policies do in those cells is read from the runs.

*11/27 08:54 KST: the reference run of the thermal rule is queued before the fourth pilot run has ended.* Three of the
four pilot runs of `V2_place` have ended: plain rgbdt 0.514, plain rgbd 0.505, `cue` 0.784 (macro average of the 11
cells, 1320 episodes each; `readout_pilot.txt` is written when the fourth has ended). Cell 9 (thermal faint alone) is
below 0.30 in all three, 0.092, 0.067 and 0.158; no other cell is below 0.30 in all three (cell 4, thermal alone:
0.058, 0.067, 0.458). The fourth run, the one network, has not started: it waits for two GPUs that are lent out and
ends at about 20:00 today. The rule above adds `V2_place_rgbt_430demos_thermal_imagenet` when a cell is below 0.30 in
all four runs. The run is queued now instead, behind the runs that decide the freeze and before the remaining runs
of the first version, so that it ends at about 17:30 today and not in the night after the fourth run. The rule itself
is as written: the reference decides something only if the one network is below 0.30 in cell 9 as well; if that run
solves the cell, the reference is reported and nothing follows from it. What follows from the reference is fixed in
point 3 above and is not chosen after its result. Demos: `results/data/sim_demos_V2_place_430_thermal.zarr`, the 430
demos of cells 0, 1, 2, 4, 9, 10 of the 700 (`python policy/subset_demos.py results/data/sim_demos_V2_place_700.zarr
results/data/sim_demos_V2_place_430_thermal.zarr --cells 0 --cells 1 --cells 2 --cells 4 --cells 9 --cells 10`: 140,
70, 70, 50, 50, 50). Command: that of the plain runs of the pilot with `ZARR=` this set, `TAG=thermal_imagenet` and
`bash scripts/run_experiment.sh V2_place rgbt 430`; the configuration of the run was printed and checked (colour and
thermal images, no depth), no start test was run because four jobs were running. Not fixed in advance: the time at
which the run is queued. A run that fails is reported and not started again.

*11/27 20:00 KST: the reference run has ended (19:56 KST); the fourth pilot run has not (about 23:00).*
`V2_place_rgbt_430demos_thermal_imagenet`, 1320 episodes: macro average 0.133, glass picked 0.197; cells 0 to 10: 0.133,
0.158, 0.167, 0.183, 0.142, 0.108, 0.108, 0.167, 0.083, 0.108, 0.100. In cell 9 it is 0.108, below 0.30. Its training
ended as that of the plain runs (101 epochs, training loss 0.0013 as plain rgbdt). The verdict is not given yet: by
point 2 it needs all four pilot runs. `python policy/readout_v2_thermal_rule.py` prints the table and applies points
2 and 3 as written (the script is committed with this note, before the fourth run has ended; its output is saved as
`readout_thermal_rule.txt` when that run has ended). If the one network is below 0.30 in cell 9, the cell counts as
not solvable and point 3 follows; if it solves the cell, nothing follows.

Next to it, not fixed in advance: the reference is below 0.30 in every cell, also where thermal shows the glass fully
(cells 0, 1, 2, 4: 0.133 to 0.167), also where the colour camera shows it too (cells 0 and 2), and it is as high in
cells it has no demos of and in which neither of its sensors shows the glass (cell 5 0.108, cell 8 0.083). A plain
policy without depth does not read the glass from the thermal image at the real contrast in either state; what it
reaches is the same everywhere. So this reference says little about the faint state by itself: in cell 4 (thermal
alone, shows) it is at 0.142 where `cue` is at 0.458. By point 2 cell 4 stays as it is, because a run solves it. The
rule is not changed for this; what the reference can and cannot show is to be said with the verdict.

**The repeated pilot, prepared before the verdict (11/27 20:04 KST; no demo of the new level collected, no cue fitted, no
policy trained; the fourth pilot run of `V2_place` ends at about 23:30 KST).** Cell 9 is below 0.30 in three of the
four pilot runs and in the reference. If the one network is below 0.30 there as well, point 3 of the rule follows, and
what it needs is prepared now so that the runs can start with the verdict. None of it is queued or reported as a
result unless the verdict is "not solvable"; if the one network solves cell 9, the level below stays unused and is
taken out of the manifest again.

- Level `V2b_place` ([`sim/configs/V2b_place.yaml`](../../../sim/configs/V2b_place.yaml), id 43, base `V2_place.yaml`):
  only the two thermal ranges differ, `faint_contrast: [0.36, 0.92]` and `show_contrast: [0.92, 3.66]`.
  `python sim/measure_visibility.py V2b_place -n 440`
  ([`measurements/sim_V2b_place_visibility.txt`](measurements/sim_V2b_place_visibility.txt)), p10 / median / p90 in
  units of the camera noise: thermal shows 32.6 / 66.2 / 94.9 (the range of point 3 is 26 to 103; a uniform draw from
  it gives 34 / 65 / 95), faint 10.5 / 18.2 / 23.9 (10 to 26; 12 / 18 / 24), blind 0.4 / 1.5 / 3.4 as on `V2_place`.
  The rows of the colour camera and of depth are identical to those of `V2_place`. `V2_place` itself is not changed,
  so its runs stay as they are; the manifest has one more entry and is otherwise identical (the parsed files compared).
- The four runs, seed 42, the protocol and the commands of step 3 with the level `V2b_place`:
  `V2b_place_rgbdt_700demos_imagenet`, `V2b_place_rgbd_700demos_imagenet`, `V2b_place_rgbdt_700demos_hcg2_cue`,
  `V2b_place_rgbdt_700demos_hcg2_cue_early`. Before them, as in step 3: 700 demos
  (`results/data/sim_demos_V2b_place_700.zarr`) and the three cue fits (`data/hcue/V2b_place_700`, `..._early`,
  `..._release`). The demos and the fits are started now, the fits beside the four jobs that run (small networks on
  2800 frames; stopped if they need more than 8 GB of memory). Through the queue of the fusion-vis worktree, before
  the lines of the first version that have not started. A run that fails is reported and not started again.
- Read-out, fixed now: `python policy/readout_v2_pilot_b.py > readout_pilot_b.txt`, the read-out of the pilot
  unchanged, on `V2b_place` and next to it on `V2_place`.
- What follows from it, fixed now: the rule changes the thermal states once. If on `V2b_place` no cell is below 0.30
  in all four runs and not every cell of every run is above 0.95, the thermal states of `V2b_place` are those of the
  benchmark for the three tasks and the freeze (step 5) follows. If a cell is again below 0.30 in all four runs, or
  everything is above 0.95, the stop of step 3 applies with the user; nothing more is changed by rule.
- Left to the user at the freeze, not fixed here: the levels of T2 and T3 take the same thermal states and their
  demos are collected again for the full run. Proposed: their plain runs on the earlier states are not repeated
  before the freeze. They answered whether the task can be learned (the stop of step 4), and a clearer thermal image
  does not make a task harder to start on.

*11/27 21:41 KST: the demos and the cues of `V2b_place` are there; nothing is queued.* 700 demos collected 20:05 to 20:15 KST
(the teacher succeeds in 140 of 140 for each of the five glasses); the three fits 20:15 to 21:35 (maps later than the
demos; at most 2.4 GB of memory beside the four jobs), the cue at reset on the two test sets until 21:40
(`data/runs/v2b_prepare.sh`, log `data/hcue/v2b_prepare.log`). The same numbers as for `V2_place` above, rule
`consensus`, with those of `V2_place` in brackets:

- grasp cue: within 1.5 cm of the hindsight grasp point in 0.85 of the 700 demos (0.82). At reset within 1.5 cm of
  the glass in 0.85 of `j_cells11` and 0.81 of `k_cells11_heldout` (0.81, 0.78): cell 4 (thermal alone) 0.82 and 0.68
  (0.60, 0.60), cell 9 (thermal faint alone) 0.22 and 0.20 (0.05, 0.03), cell 8 (depth faint alone, not changed) 0.63
  and 0.52 (0.65, 0.48);
- place cue: within 1.5 cm in 0.96 of the demos (0.97);
- the one network (`V2b_place_700_early`): 0.79 of the demos (0.80); at reset 0.79 and 0.78 (0.77, 0.74), cell 4 0.28
  and 0.30 (0.27, 0.15), cell 9 0.13 and 0.08 (0.07, 0.03);
- own peak of the thermal sensor within 1.5 px where it shows the glass / faintly / not: 0.82 / 0.41 / 0.09 (0.70 /
  0.21 / 0.12); colour and depth as before (0.96 / 0.91 / 0.15 and 1.00 / 0.90 / 0.16).

Fits only, and nothing is decided from them. With the thermal states of point 3 the cue finds the glass of cell 9 in
about a fifth of the scenes, where it found it in a twentieth; whether a policy gets over 0.30 there is read from the
four runs, if the verdict calls for them.

**Step 4, the results on T2 and T3 (11/27 21:42 KST; the three runs have ended: T3 plain 11/27 12:20 KST, T2 plain 16:34, the
method on T2 21:41).** [`readout_tasks.txt`](readout_tasks.txt), seed 42.

| task | run | macro average | glass picked | lowest and highest cell |
|---|---|---|---|---|
| T2 `V2_align`, 1320 episodes | plain rgbdt | 0.564 | 0.792 | 0.233 (cell 9), 0.808 (cell 5) |
| | `cue` with the angles | 0.828 | 0.965 | 0.417 (cell 9), 0.933 (cells 1, 10) |
| T3 `V2_two`, 720 episodes | plain rgbdt | 0.143 | 0.471 | 0.017 (pair 3), 0.333 (pair 1) |

- The stop of step 4 applies to neither task (plain rgbdt 0.564 and 0.143, both above 0.05; the best cell of a run
  0.933 and 0.333). Both tasks can be started on.
- T2: the method against plain rgbdt on the same episodes: only the first succeeds in 413, only the second in 65
  (p = 5.6e-63); it is higher in each of the 11 cells. Failures of the plain run: the glass put down off the mat
  (219), not grasped (131); of the method: off the mat (140). Angle between glass and mat at the end, glass picked:
  median 4.0 degrees and 3.3 degrees, within 15 degrees in 0.92 and 0.97.
- T3: one glass on a coaster in 0.442 of the episodes, both in 0.143; no other failure label than "one placed" (402)
  and "none placed" (215). The method is not run on T3 (no cue for two targets yet).

Next to it, not fixed in advance: (1) on T2 the method is at 0.417 in cell 9 (thermal faint alone) with the thermal
states of `V2_place`, where on T1 it is at 0.158; plain rgbdt is at 0.233 there (T1: 0.092). (2) Plain rgbdt on T2 is
at 0.23 to 0.49 in the cells without depth (2, 4, 6, 7, 9), on T1 at 0.06 to 0.13, and lower than on T1 where depth
shows (0.77 to 0.81 against 0.98 to 0.99). The first is in the direction of what the layout gives on T2 (below, "What
a blind sensor still shows"); the tasks and the glasses differ, so it is not shown by this. One seed.

**The verdict of the rule: cell 9 is not solvable, the pilot is repeated once on `V2b_place` (11/27 23:50 KST; the fourth pilot
run ended 11/27 23:46 KST; no run on `V2b_place` has started and none is in the queue yet).**
[`readout_pilot.txt`](readout_pilot.txt) and [`readout_thermal_rule.txt`](readout_thermal_rule.txt), both saved 11/27
23:47 KST; `V2_place`, seed 42, 1320 episodes per run.

| cell | sensors that show the glass | plain rgbdt | plain rgbd | `cue` | one network | reference (plain rgbt, 430 demos) |
|---|---|---|---|---|---|---|
| 0 | rgb+depth+thermal | 0.975 | 0.967 | 0.975 | 1.000 | 0.133 |
| 1 | depth+thermal | 0.983 | 0.983 | 0.950 | 0.967 | 0.158 |
| 2 | rgb+thermal | 0.108 | 0.092 | 0.900 | 0.808 | 0.167 |
| 3 | rgb+depth | 0.992 | 0.975 | 0.967 | 1.000 | 0.183 |
| 4 | thermal | 0.058 | 0.067 | 0.458 | 0.217 | 0.142 |
| 5 | depth | 0.983 | 0.992 | 0.950 | 0.983 | 0.108 |
| 6 | rgb | 0.133 | 0.058 | 0.892 | 0.825 | 0.108 |
| 7 | rgb (faint) | 0.067 | 0.075 | 0.842 | 0.733 | 0.167 |
| 8 | depth (faint) | 0.633 | 0.617 | 0.617 | 0.642 | 0.083 |
| 9 | thermal (faint) | 0.092 | 0.067 | 0.158 | 0.067 | 0.108 |
| 10 | all three (faint) | 0.633 | 0.658 | 0.917 | 0.892 | 0.100 |
| macro average | | 0.514 | 0.505 | 0.784 | 0.739 | 0.133 |
| glass picked | | 0.570 | 0.534 | 0.840 | 0.792 | 0.197 |

- The stops of step 3: one cell is below 0.30 in all four runs, cell 9 (thermal faint alone; the best run is `cue`
  at 0.158). Not everything is above 0.95.
- Point 2 of the rule: the reference is at 0.108 in cell 9, so the cell counts as not solvable. Cell 4 (thermal alone,
  shows) stays as it is: `cue` solves it (0.458). What the reference can show was said at 20:00 KST above: it is
  below 0.30 in every cell, so by itself it says little about the faint state. The rule is applied as written.
- Point 3: cell 9 is a cell in which thermal is faint. Thermal "faint" becomes 10 to 26 noise units and "shows" 26 to
  103, which is the level `V2b_place` prepared above, and the four pilot runs are repeated once on it. This is the one
  change the rule makes; `V2_place` and its runs stay as the record of the first pilot.
- `cue` against plain rgbdt on the same episodes: 0.784 against 0.514, only the first succeeds in 417, only the
  second in 61 (p = 3e-66). `cue` against the one network: 0.784 against 0.739, 144 and 85 (p = 0.00012). Plain rgbdt
  against plain rgbd: 0.514 against 0.505, 71 and 58 (p = 0.29). One seed; by its listing the pilot decides the states
  and nothing of it enters the tables of the frozen benchmark.

Next to it, not fixed in advance: per cell `cue` is above the one network in six cells (2, 4, 6, 7, 9, 10; the widest
gap in cell 4, thermal alone, 0.458 against 0.217) and below it in five (0, 1, 3, 5, 8, by 0.017 to 0.033). The five
are cells in which depth shows the glass (fully in 0, 1, 3 and 5, faintly in 8), and in them all four runs are within
0.05 of each other. Both runs with a cue trained to the end (epoch 100, training loss 0.0006; plain rgbdt 0.0013).

*The repeated pilot in the queue.* The four runs are put in the queue of the fusion-vis worktree after this commit,
before the lines of the first version that have not started, with the commands of step 3, the level `V2b_place` and
its cue folders (`data/hcue/V2b_place_700`, `..._early`, `..._release`), in this order:
`V2b_place_rgbdt_700demos_hcg2_cue`, `V2b_place_rgbdt_700demos_hcg2_cue_early`, `V2b_place_rgbdt_700demos_imagenet`,
`V2b_place_rgbd_700demos_imagenet`. The four GPUs hold runs of the first version (the queue started its next line on
the GPU of the fourth pilot run in the minute that run ended, before the verdict was read). They are not stopped;
the four runs start as these end, at about 01:40, 03:20, 03:45 and 05:20 KST of 11/28, and end between about 09:30
and 13:30 KST.

*When the outcome is known, said before any of them starts.* What follows is as fixed at 20:04 KST above and is read
from `readout_pilot_b.txt` when all four have ended. Both stops can be ruled out by one run alone: if a run has
every cell at 0.30 or more and one cell at 0.95 or below, then no cell is below 0.30 in all four runs and not
everything is above 0.95, whatever the other three give. A run can be read this way as soon as it has ended. The
method's run goes first because on `V2_place` it is the highest run in the cells without depth, so it is the one
most likely to settle it early. The order changes when the outcome is known, not what it is, and all four runs are
run and reported in either case. If the first run does not settle it, nothing is said before all four have ended.

*11/28 02:16 KST: one GPU goes to another project of the user; the times of the repeated pilot change, nothing else.* The
method's run started 11/28 01:20 KST on GPU 1 (20 minutes before the time given above). At 02:16 KST the queue is set
to three runs at once on GPUs 0, 1 and 3; GPU 2 is handed over when the run of the first version on it ends (about
03:10 KST). The one network starts when GPU 3 is free (about 03:30), plain rgbdt when GPU 0 is (about 05:00), plain
rgbd when the method's run has ended (about 09:00). Ends: the method about 09:00 to 09:30, the one network and plain
rgbdt about 11:30 to 13:00, plain rgbd about 16:30 KST. The order, the commands and the read-out are as written.

*11/28 09:03 KST: the method's run on `V2b_place` has ended; it does not settle the outcome early.*
`V2b_place_rgbdt_700demos_hcg2_cue` trained and was evaluated as listed (11/28 01:20 to 09:02 KST). Macro average
0.815 (0.784 on `V2_place`). Cell 9 is at 0.217 (26 of 120 episodes; 0.158 on `V2_place`), cell 4 (thermal alone)
at 0.683 (0.458), every other cell at 0.63 or more and six of them (2, 4, 6, 7, 8, 10) at 0.95 or below. A run with a cell below 0.30
rules out neither stop by itself, so, as said before the runs started, the outcome is read from
`readout_pilot_b.txt` when all four have ended. Cell 9 is not "below 0.30 in all four runs" if one of the other
three is at 0.30 or more in it; on `V2_place` these three were at 0.067 to 0.092 there. The other runs: the one
network started 03:32 KST, plain rgbdt 04:55 KST, plain rgbd 09:02 KST on the GPU of the method's run (ends about
16:30 to 17:00 KST). Said to the user before the end of the run, on a direct question about the state of the
benchmark (08:33 KST): the number of cell 9 from the first of the two test sets (0.217, 60 episodes). Nothing was
decided from it.

**The repeated pilot: cell 9 is again below 0.30 in all four runs; the stop applies, with the user (11/28 14:40 KST; the
four runs ended 09:02, 11:18, 12:41 and 14:37 KST of 11/28, none failed).**
[`readout_pilot_b.txt`](readout_pilot_b.txt), the fixed command, saved 11/28 14:39 KST; `V2b_place`, seed 42, 1320
episodes per run, the numbers of `V2_place` in brackets.

| cell | sensors that show the glass | plain rgbdt | plain rgbd | `cue` | one network |
|---|---|---|---|---|---|
| 0 | rgb+depth+thermal | 0.975 (0.975) | 0.983 (0.967) | 0.983 (0.975) | 0.992 (1.000) |
| 1 | depth+thermal | 1.000 (0.983) | 0.975 (0.983) | 0.958 (0.950) | 0.992 (0.967) |
| 2 | rgb+thermal | 0.175 (0.108) | 0.092 (0.092) | 0.925 (0.900) | 0.833 (0.808) |
| 3 | rgb+depth | 1.000 (0.992) | 0.983 (0.975) | 0.983 (0.967) | 0.983 (1.000) |
| 4 | thermal | 0.083 (0.058) | 0.025 (0.067) | 0.683 (0.458) | 0.342 (0.217) |
| 5 | depth | 0.992 (0.983) | 0.975 (0.992) | 0.967 (0.950) | 0.983 (0.983) |
| 6 | rgb | 0.108 (0.133) | 0.067 (0.058) | 0.858 (0.892) | 0.858 (0.825) |
| 7 | rgb (faint) | 0.075 (0.067) | 0.067 (0.075) | 0.850 (0.842) | 0.767 (0.733) |
| 8 | depth (faint) | 0.625 (0.633) | 0.583 (0.617) | 0.633 (0.617) | 0.675 (0.642) |
| 9 | thermal (faint) | 0.142 (0.092) | 0.067 (0.067) | 0.217 (0.158) | 0.092 (0.067) |
| 10 | all three (faint) | 0.575 (0.633) | 0.642 (0.658) | 0.908 (0.917) | 0.908 (0.892) |
| macro average | | 0.523 (0.514) | 0.496 (0.505) | 0.815 (0.784) | 0.766 (0.739) |
| glass picked | | 0.571 (0.570) | 0.533 (0.534) | 0.870 (0.840) | 0.814 (0.792) |

- The stops, as fixed at 11/27 20:04 KST: one cell is below 0.30 in all four runs, cell 9 (thermal faint alone; the
  best run is `cue` at 0.217). Not everything is above 0.95. So the first condition for the freeze is not met:
  **the stop of step 3 applies with the user, and nothing more is changed by rule.** The thermal states of
  `V2b_place` are not the benchmark's by this read-out, and nothing is frozen.
- What the one change of the rule did: cell 4 (thermal alone, shows) went from 0.458 to 0.683 for `cue` and from
  0.217 to 0.342 for the one network; cell 9 from 0.158 to 0.217 and from 0.067 to 0.092. The cells in which
  thermal does not show the glass (3, 5, 6, 7, 8) are where they were (within 0.04).
- `cue` against plain rgbdt on the same episodes: 0.815 against 0.523, only the first succeeds in 443, only the
  second in 57 (p = 4.3e-75). `cue` against the one network: 0.815 against 0.766, 141 and 76 (p = 1.2e-05). Plain
  rgbdt against plain rgbd: 0.523 against 0.496, 76 and 41 (p = 0.0016; on `V2_place` 71 and 58, p = 0.29). One
  seed; a pilot, nothing of it enters the tables of a frozen benchmark.

Next to it, not fixed in advance, for the decision on cell 9 (`cue` run, the cue inside the 120 episodes of a cell;
"right" is within 1.5 cm of the glass in the table plane):

| cell | grasp cue right | success where it is right | success where it is not |
|---|---|---|---|
| 9, thermal faint alone | 0.21 (25 episodes; median error 4.2 cm) | 0.68 | 0.09 |
| 4, thermal alone | 0.75 (median 0.8 cm) | 0.79 | 0.37 |
| 8, depth faint alone | 0.58 (median 1.3 cm) | 0.86 | 0.33 |
| 7, rgb faint alone | 0.93 (median 0.4 cm) | 0.89 | 0.33 |

- In cell 9 the policy does what it does elsewhere once the glass is located (0.68 of 25 episodes); what is missing
  is the location, in four scenes of five. The same holds for the one network (its map is right in 0.11 of the
  episodes; success 0.31 there and 0.07 elsewhere).
- Within cell 9 the success of `cue` rises with the thermal contrast of the scene: 0.15, 0.18 and 0.33 in the
  thirds of the contrast range (40 episodes each). 47 of the 120 scenes are solved by at least one of the four runs.
- In cell 9 alone: `cue` against plain rgbdt 20 and 11 (p = 0.15), against the one network 22 and 7 (p = 0.008).
- All four runs are above 0.95 in the cells in which depth shows the glass fully (0, 1, 3, 5) and nowhere else.

So the evidence on cell 9 is: at the real p10 to p25 of the thermal contrast no run reaches 0.30; the glass is in the
thermal image (section 11: inside the simulator the window with the glass is told from the one without at 0.93 already
at 3 to 10 units), a policy that is given its location grasps it, and the networks tried here locate it in a fifth of
the scenes. Whether a better thermal network locates it more often was not tried. The reference of point 2 was no
witness either way (below 0.30 in every cell).

*For the user: what is done with cell 9.* Nothing below is started or changed before the decision.

1. **Keep cell 9 at the states of `V2b_place` and freeze, with the cell marked as open** (the proposal). The tables
   carry all eleven cells and the macro average over eleven; the text says that no policy of the pilot is at 0.30 in
   cell 9 and gives the table above. For: the state is the measured real range and is not chosen by what policies
   reach; the cell is not empty of information; it costs no further run. Against: the benchmark is frozen with a
   cell for which no run shows that it can be solved, which is what the stop of step 3 was written to prevent.
2. **Take cell 9 out: ten cells.** Thermal "faint" then stands only in cell 10 (all three faint). For: every cell
   left is solved by a run. Against: a condition that occurs on real glass is dropped because it is hard, and the
   three sensors are no longer treated alike (colour and depth keep their "faint alone" cells).
3. **Raise thermal "faint" once more and repeat the four runs** (about 15 hours on three GPUs, 30 GPU hours). The
   next range would be above the real p25, which is where "shows" begins on `V2b_place`; the states would leave what
   the rule of 11/26 allowed, and the pilot would have been repeated twice on the cell the method is best in.

In each case `V2b_place` against `V2_place` has to be decided with it: under 1 and 2 the proposal is the thermal
states of `V2b_place` (cell 4 is solved more clearly there, 0.683 against 0.458, and both lie in the measured range).

**What a blind sensor still shows: the layout (11/26 19:51 KST; looked at when the fits of T2 were read, not a test fixed in
advance).** In the report of the T2 cue the sensors' own maps are on the grasp pixel also where the sensor does not
show the glass: within 1.5 px in 0.40 of the colour-blind demos, 0.29 of the depth-blind and 0.23 of the
thermal-blind ones. On `L0_vis` these are 0.10, 0.13 and 0.08, and the same peaks paired with the grasp pixels of
other demos give 0.07. Two checks (`policy/check_blind_layout.py`, output in [`blind_layout.txt`](blind_layout.txt)):

1. *The frames.* At the policy's 96 x 96 the colour-blind frame of `V2_align` with the glass and with the glass
   taken out of the scene differ in 0.05 of 130 scenes, by one pixel and one grey level, never within 3 px of the
   glass; the depth-blind frames in 0.06 of 128 scenes (two pixels, at most 3 of 255). `V2_place` and `L0_vis` are
   the same. The glass is not in these images.
2. *The layout.* The generator keeps every distractor 9 cm (edge to edge) away from the glass and the place target
   clear of it, and the glass spawns in a region of 12 x 30 cm. With one to three things on the table besides the
   place target, that leaves little: the spawn points at which the glass can be are a median 0.13 of the region on
   `V2_align`, 0.25 on `V2_place` and 0.38 on `L0_vis`. Their centroid, computed from the positions of the
   distractors and the place target and from no image, is a median 3.1 cm from the glass on `V2_align` (within 3 cm
   in 0.49 of 700 scenes), 3.8 cm on `V2_place` (0.37) and 4.6 cm on `L0_vis` (0.27). The centre of the spawn region
   is 8.5 cm away (0.08).

So a sensor that is blind to the glass still shows where the glass is not, and a network turns that into a position
a few centimetres off. Consistent with it, the three sensors are in the order of how much of the layout each shows
(colour all of it, depth less of the flat mat, thermal only what differs in temperature). It is a property of the
scenes in both versions of the benchmark, weaker in the first (0.07 to 0.15 on `L0_vis` and `L0_vis_place`) and
strongest on T2, where the mat and the glass are large. It is open to every policy alike, baseline or method, and it
is not information from the simulator: a camera sees the same in a real scene that is laid out this way. What it
changes is how a cell is read. "Colour is blind" means that the glass is not in the colour image, not that the
colour image says nothing about where to grasp; and in the cells whose only sensor is faint (7, 8, 9) part of a
policy's success can come from the layout.

Nothing is changed. A clearance of this order is what the open gripper (14 cm wide) needs to come down around the
glass; how small it could be was not tested, and a larger spawn region or more distractors would change the scenes of
every level, the frozen ones included. Open, for the user: whether the scenes stay as they are (the proposal), with a
reference per task that gets no glass in any image, so that the share of the layout is a number in the tables.

**Still open for the freeze.** T2 has three training shapes, so a number of demos that is a multiple of 210 keeps the
cell weights exact (630 or 840; 700 gives 76 / 70 / 68 demos in cells 1 to 3). T3: the cue of the method for two
targets. The plain runs of T2 and T3 and the solvability checks.

## 11. Check against real sensor images: is the glass as hard to find at the same visibility?

**Why.** Section 2 matched how much of the glass a sensor shows (a contrast, a share of holes). It did not look at
what the glass looks like: the renderer does not refract or reflect, the holes of the depth model are single pixels
drawn at random, the thermal model reflects one uniform room. If glass of a given contrast is much easier to find in
the simulator than in real images, the cells are easier than their numbers say. This can be measured without a robot.

**Listed 11/26 11:21 KST, before any patch is cut or any classifier trained.** Only the sizes of the glass regions
and how many of them get a pair of windows were looked at (to fix the geometry below).

*Unit.* A glass region of a frame (a connected part of the glass mask, `real_data/glass_visibility.py`) and its
visibility number per sensor, as in section 2. Around it a square window of 1.5 times the longer side of its bounding
box, and a background window of the same size in the same frame, its centre 1.0 to 2.5 window sides away, with at
most 5% glass in it (60 draws; a region without such a pair is left out and counted). Both are resized to 32 x 32.
A nearby background keeps the surroundings alike, so that what tells the two windows apart is the glass.

*Images.* Real, object-like glass only: TRansPose (colour, lidar depth, thermal; 126 frames drawn from each of its 35
sequences, seed 0) and T-ROBI (hole-type depth; all 660 frames). The panes of the other three datasets are not
objects on a table and are left out. Simulator: the first frames of 3000 episodes of `V2_place` (seeds 0 to 2999) at
the sensors' own resolution, with the true glass mask.

*Inputs, the same for both.* Colour: three channels in 0 to 1. Depth: (reading - median reading of the window) / 10
cm, clipped to +-1, zero where there is no reading, and a second channel that marks the pixels without a reading.
Thermal: (raw count - median of the window) / the camera's noise (0.89 counts), divided by 100 and clipped to +-1.

*Classifier.* Per sensor and per domain one small network (three 3 x 3 convolutions of 32, 64, 64 channels with
batch norm and pooling, a linear layer), Adam 1e-3, 30 epochs, batch 256, flips and shifts of 2 pixels, seed 0. Scores
out of fold: five folds over the sequences (real) or over the episodes (simulator). For the transfer, one network
trained on all pairs of one domain scores all pairs of the other.

*Bins, by the region's visibility number (the ranges of section 3).* Thermal, noise units: below 3, 3 to 10, 10 to
30, 30 to 103, above 103. Colour, grey levels: below 3, 3 to 10, 10 to 20, 20 to 45, above 45. Depth: hole (0.5 or
more of the pixels missing), partial (0.05 to 0.5), see-through (below 0.05).

**Read-out, fixed now** (`python real_data/findability.py readout > results/hard_scenario/bench_v2/real_check/readout.txt`).
Per sensor and bin: the number of pairs and the AUC (glass window against its background window) in real images and
in the simulator, each with a 95% interval (1000 resamples of the sequences or episodes), and their difference.

1. *The floor of the test.* In the simulator's blind states the glass is not in the image at all (colour blind, depth
   see-through, thermal below 3), so the AUC there should be 0.5. Whatever it is, it is what the windows give away
   without the glass, and the other numbers are read against it.
2. *Same difficulty.* A bin with at least 50 pairs in both domains is "the same" if the difference is within 0.05 or
   its interval contains 0; "easier in the simulator" or "harder in the simulator" otherwise.
3. *Does the look transfer?* The network trained in the simulator scores the real pairs (and the reverse). In a bin
   in which the real network reaches at least 0.60, the look "transfers" if the simulator's network keeps at least
   half of the real network's margin over 0.5.

**What it can and cannot say, stated before the numbers.** Real backgrounds are more varied than the simulator's
table, so a lower AUC in real images is not the sensor model's fault alone; a transfer that fails does not say why.
The regions that get a pair are the smaller, free-standing ones. Objects of TRansPose recur across its sequences. It
is one real camera per kind of sensor. Finding a window is not a policy's success: this check does not replace a
real robot. A bin that is easier in the simulator is reported with what could be changed in the sensor model, as a
proposal to the user before the freeze; nothing is changed by the read-out itself.

**Result of the check as listed (11/26 11:30 KST): it fails its own control, and nothing is concluded from it.**
[`real_check/readout.txt`](real_check/readout.txt). Real: 3050 pairs (TRansPose 1888 of 8699 regions, T-ROBI 1162 of
1960; the others have no window that fits or no background window). Simulator: 3000 pairs. The first start stopped at
the import of a package that is not installed, before any classifier was trained; the folds are made by a few lines
of the script instead (the groups, largest first, each to the fold with the fewest pairs).

| the floor (rule 1): the simulator's blind state, the glass is not in the image | pairs | AUC |
|---|---|---|
| colour, contrast below 3 grey levels | 1162 | 0.99 |
| depth, see-through | 1165 | 0.99 |
| thermal, below 3 noise units | 1015 | 0.95 |

The classifier tells the glass window from the background window where there is no glass to see: in the simulator
the windows give the answer away. Looked at afterwards (next to the read-out, not a test): in the colour-blind pairs
the glass window is the busier one (standard deviation of its pixels 0.148 against 0.074 in the median; that number
alone gives an AUC of 0.74). The glass stands in the part of the scene where the coaster, the distractors and the arm
are, and a window one to two sides away is plainer. So the simulator's AUC is 0.95 to 1.00 in every bin whatever the
visibility, and the verdicts of rule 2 ("easier in the simulator" for colour, "the same" for depth and thermal) say
nothing about the sensor model. They are in the file and are not read.

What the run does show: (a) a network that learned the simulator's windows scores real windows at 0.91 where the
depth is a hole and near or below chance elsewhere (0.14 on real see-through depth, 0.12 on real thermal above 103
units): it learned the simulator's surroundings, which are the other way round in the real scenes; (b) in real images
the same small network reaches 0.72 to 0.91 per colour bin (0.84 overall), 0.97 to 0.99 for depth and 0.93 to 1.00
for thermal, each an upper bound, since real windows may give something away as well.

**The check, second design: in the simulator the same window with and without the glass (listed 11/26 11:33 KST,
before any of it is run).** The surroundings must be the same in the two windows. In the simulator that can be had
exactly: the first frame is rendered twice, as drawn and with the glass moved out of the scene (10 m away; nothing
else changes; the sensor noise is drawn anew), and the pair is the window around the glass in the first render and
the same window in the second. Real images have no such pair, so the real pairs stay as they are (already cut), and
the two domains are no longer compared by their own AUC: a paired task in one and an unpaired one in the other are
not the same task. What is read is whether the look carries over between the domains.

*Runs.* `python real_data/findability.py simpair` (3000 episodes of `V2_place`, seeds 0 to 2999, the windows, inputs
and bins of the first design), `... fit --sim simpair`, `... readout2 > real_check/readout2.txt`. Networks and
training as in the first design; the simulator's network is trained on the paired windows.

**Read-out, fixed now.** Per sensor and bin: the simulator's own AUC out of fold (with against without the glass);
the real network on the simulator's pairs, as an AUC and as the share of pairs in which the window with the glass
scores higher; the simulator's network on the real pairs; the real network's own AUC out of fold (from the first
design's real pairs).

1. *Floor.* Colour-blind (below 3 grey levels) and depth see-through pairs of the simulator are the same image twice
   up to the sensor noise: the simulator's own AUC must be within 0.45 to 0.55 there. If it is not, nothing else is
   read. Thermal "blind" is not part of the floor: it is a contrast of up to 0.1 degC (below 3 noise units per pixel),
   which a network that averages over the glass may see; what the simulator's network reaches there is a result
   (is thermal blind blind to a network?).
2. *Does a network that learned real glass respond to the simulator's glass?* In a bin with at least 50 pairs in
   both domains in which the real network's own AUC is at least 0.60: "yes" if the real network's AUC on the
   simulator's pairs is at least 0.5 + half of the real network's own margin over 0.5; "no" otherwise.
3. *Does a network that learned the simulator's glass find real glass?* The same bins and the same threshold for the
   simulator's network on the real pairs: "transfers" or "does not transfer".
4. The simulator's own AUC next to the real one is printed for orientation and is not a verdict, for the reason
   above.

**What it can and cannot say.** A "yes" in 2 says that the simulator's glass moves a real-glass network's score the
right way, not that it looks like real glass in every respect; a "no" in 2 or 3 can also come from the scenes around
the glass, which differ. The limits listed with the first design hold. It does not replace a real robot.

**Result of the second design (11/26 11:41 KST).** [`real_check/readout2.txt`](real_check/readout2.txt); 3000 pairs of
the simulator, the 3050 real pairs of the first design.

*Rule 1, the floor: holds.* Colour blind 0.50 (1162 pairs), depth see-through 0.50 (1166 pairs): with the same window
rendered with and without the glass, nothing but the glass tells the two apart.

| sensor, bin | simulator's own AUC | real network's own AUC (an upper bound) | real network on the simulator's pairs: AUC; share of pairs with the glass scored higher | rule 2: does a real-glass network respond? | simulator's network on real pairs | rule 3 |
|---|---|---|---|---|---|---|
| depth, hole | 1.00 | 0.99 | 0.92; 1.00 | **yes** | 0.83 | **transfers** |
| depth, partial (42 real pairs) | 1.00 | 0.83 | 0.47; 0.39 | not judged | 0.81 | not judged |
| depth, see-through | 0.50 | 0.97 | 0.50; 0.03 | no | 0.34 | does not transfer |
| thermal, below 3 units (35 real pairs) | 0.66 | 0.92 | 0.50; 0.48 | not judged | 0.45 | not judged |
| thermal, 3 to 10 | 0.93 | 0.92 | 0.51; 0.59 | no | 0.55 | does not transfer |
| thermal, 10 to 30 | 0.99 | 0.98 | 0.59; 0.87 | no | 0.29 | does not transfer |
| thermal, 30 to 103 | 1.00 | 0.99 | 0.80; 0.98 | **yes** | 0.36 | does not transfer |
| thermal, above 103 (42 simulator pairs) | 1.00 | 1.00 | 0.88; 1.00 | not judged | 0.54 | not judged |
| colour, below 3 grey levels | 0.50 | 0.74 | 0.50; 0.00 | no | 0.59 | does not transfer |
| colour, 3 to 10 | 1.00 | 0.72 | 0.48; 0.38 | no | 0.49 | does not transfer |
| colour, 10 to 20 | 1.00 | 0.83 | 0.51; 0.53 | no | 0.48 | does not transfer |
| colour, 20 to 45 | 1.00 | 0.78 | 0.46; 0.24 | no | 0.52 | does not transfer |
| colour, above 45 | 1.00 | 0.91 | 0.43; 0.09 | no | 0.64 | does not transfer |

- **Depth, the hole: the look carries over both ways.** A network that learned real holes scores the simulator's
  window with the glass higher in every pair, and the simulator's network finds real holes (0.83). The state "depth
  shows" of the second version is supported.
- **Thermal: a network that learned real glass responds to the simulator's glass at the contrast of "shows"** (30 to
  103 units: 0.80, the window with the glass higher in 0.98 of the pairs). At 10 to 30 units the rule says no (0.59
  against the 0.74 it asks for); next to the read-out: the window with the glass is still the higher one in 0.87 of
  the pairs. The AUC ranks windows of different episodes against each other, and the real network's score differs
  more between scenes than a weak glass moves it. At 3 to 10 units ("faint") it does not respond (0.51; 0.59 of the
  pairs). The other way round nothing transfers, and at 10 to 103 units the simulator's network scores real glass
  below the real background (0.29, 0.36): it learned "a patch that differs from the table", and real scenes have more
  of those next to the glass than on it. That is a difference of the scenes, and this test cannot take it apart from
  the look of the glass.
- **Colour: no in every bin, both ways.** A network that learned real glass does not respond to the simulator's
  glass; at a contrast of 20 grey levels and more it scores the window with the simulated glass lower than the same
  window without it (the glass higher in 0.24 and 0.09 of the pairs). The simulator's network does not find real
  glass (0.48 to 0.64). Section 2 said the renderer does not refract or reflect, so that glass does not look like
  glass where its contrast is right; this is that limit with a number. The real colour network is also the weakest of
  the three (0.72 to 0.91 on its own pairs), so part of the "no" may be that it learned the surroundings.
- **Inside the simulator, a state that is not blind is always found in the pair**: 1.00 for colour and depth from
  the faint states on, 0.93 for thermal faint. The cells' difficulty for a policy does not come from the pair being
  hard to tell apart; it comes from finding the glass in one image.
- **Thermal "blind" is not blind in the pair: 0.66** (0.64 to 0.67, 1031 pairs). The glass at up to 0.1 degC from
  the table, averaged over its pixels, and the glass in front of something warmer or colder leave a trace. Whether a
  network that has to find the glass uses it: on `L0_vis`, which has the same blind contrast, the thermal network of
  the cue has its peak on the glass in 0.08 of the thermal-blind frames, as the colour and depth networks in their
  blind frames (0.10, 0.13), so there is no sign of it. The same number is read from the cue reports of `V2_place`
  when they are fitted.
- **Open, and this test cannot settle it: is real "blind" blind?** On real pairs the real networks reach 0.97 where
  the lidar sees through the glass, 0.92 below 3 thermal units (35 pairs) and 0.74 below 3 grey levels. These are
  upper bounds (real pairs have no floor: the surroundings of the two windows differ), but if a part of them is the
  glass, the simulator's blind states are more blind than real sensors are.

**What follows, as a proposal; nothing is changed by this.** Depth "shows" is supported. Thermal "shows" is supported
one way; thermal "faint" is not shown either way. The colour image of the glass is not supported. For the colour
image the choices are: (a) leave it and state it, as now; (b) before the freeze, try a colour model of glass in the
sensor (the background seen through it shifted as by refraction, highlights, a darker rim) and repeat this check,
which now exists and takes ten minutes; (c) leave the answer to the real robot. The user decides.

## 12. The colour image of the glass: an optical model in place of alpha

**Decision of the user (11/26): choice (b) of section 11, try a colour model of glass and repeat the check.** The
preparation of the draft levels (cue and angle fits) is stopped until this is decided: a new colour image of the
glass means new demos and new cues.

**What is wrong today.** The glass is a solid of colour (0.8, 0.9, 1.0) blended over the scene at alpha 0.5 (shows) or
0.2 (faint), lit like any other body: a pale grey solid. Real clear glass (the windows of TRansPose, looked at side
by side with the simulator's) is the background seen through it, slightly dimmed, with bright highlights on rims and
ridges and a line along its edges; there is no tint over its area.

**Listed 11/26 11:54 KST, before any of it is written or run.**

*The model* (`rgb_glass: optical`, off by default, so every existing level renders as before). The scene is rendered
without the glass (B); the glass's surface normal n is taken from its depth image; per glass pixel
`I = (1 - e) * [T0 * (1 - F) * B(p + shift) + F * E] + e * rho * E + 255 * S`, clipped, where
- `T0 = 0.92`: what two thin walls of clear glass let through (fixed);
- `F`: reflectance, Schlick with F0 = 0.08 (a wall has two surfaces of 0.04) and the angle between n and the view ray
  (fixed);
- `E`: what the glass reflects and refracts from elsewhere, taken as the mean colour of B over the image;
- `shift`: the background behind the glass moved by up to 0.15 of the glass's radius towards its edges, growing
  with the square of the sine of the angle (a thin-walled vessel bends light little in its middle; fixed);
- `e`: the edge, in 0 to 1: where the wall is seen edge-on (sine of the angle above 0.85, rising to 1), on the
  outline of the glass and on the creases between its faces; there the pixel takes `rho * E`, the dimmed
  surroundings, times a weight `w_rim`;
- `S`: highlights of the scene's lights (the camera's headlight, the lamp above, the extra lights of a lighting
  preset), Blinn-Phong with exponent 30 on the normal, times an intensity `k_s`; they follow the light level of the
  episode.
Not modelled: the inner surfaces and the far wall, the thick base, caustics and shadows on the table, liquid.
A strength s blends it with the scene, `(1 - s) * B + s * I`: s = 1 is "shows", s = 0 is "blind" (the image is B,
pixel for pixel), and "faint" is s = 0.3 for now (where that lies among real glass is measured below; ranges per
episode as for the other sensors are set before the freeze).

*Three numbers are set from real images and not from the check*: `k_s`, `w_rim` and `rho`. On 32 x 32 windows (the
geometry of section 11, with the region's mask), for real regions whose surroundings are plain (standard deviation
of the luminance outside the mask below 15 grey levels; the simulator's table is plain) and for the simulator's
"shows": the differences between the glass's pixels and the window's other pixels in the median, the 95th and the 5th
percentile of the luminance. `k_s`, `w_rim` and `rho` are chosen on a grid (k_s 0.2 to 1.6, w_rim 0 to 1, rho 0.3 to
1.0) so that the simulator's medians of the three differences are closest to the real medians (squared distance in
units of the real interquartile range). If fewer than 100 real regions have plain surroundings, the limit is raised
to 25 and this is said.

*The judges.* Real colour pairs as in section 11, cut from all 252 frames of each TRansPose sequence. The 35
sequences are split in two halves by their order (even and odd). Judge A is the network of section 11 trained on
the pairs of the even half, judge B on the odd half; a judge's own AUC is measured on the other half. While the
model is developed only judge A is looked at, in at most three rounds, each with its change and its reason written
down; judge B is run once, on the final model.

**Read-out, fixed now** (`python real_data/findability.py readout3`), on 3000 paired windows of `V2_place` with the
optical model (the same window with and without the glass):
1. *Floor.* In the colour-blind pairs the two windows are the same image: the simulator's own AUC within 0.45 to
   0.55.
2. *Does a network that learned real glass respond?* Per state (shows, faint): judge B's AUC on the simulator's
   pairs against 0.5 + half of judge B's own margin over 0.5: "yes" or "no". Next to it the share of pairs in which
   the window with the glass scores higher, and the same per contrast bin as in section 11.
3. *Does a network that learned the simulator's glass find real glass?* The simulator's network (all pairs of the
   3000) on all real pairs against 0.5 + half of the real network's own margin (out of fold): "transfers" or "does
   not transfer".
4. The alpha model through the same judges, for comparison.

**What is done with it, fixed now.** If rule 2 says "yes" for "shows", the optical model is proposed for the second
version (the user decides; the draft levels then get `rgb_glass: optical`, and their demos and cues are made again).
If it says "no" after the three rounds, the result is reported as it is, the colour image stays as it is unless the
user decides otherwise, and the limit is stated in the paper. Images of the glass under both models are shown with
the result either way.

**Result (11/26 13:07 KST): the optical model does not pass the held-out judge. By the rule above the colour image stays
as it is unless the user decides otherwise.** Files in [`real_check/`](real_check/): `look_as_listed.txt`, `look.txt`,
`look_round3.txt` (the numbers from real images), `judgeA_round1.txt` to `judgeA_round3.txt` (the development
rounds), `readout3.txt` (the final read-out), `glass_alpha_and_optical.png` (the same twelve scenes under the two
models: eight "shows", four "faint"; alpha above, optical below). The model is `sim/glass_optics.py`, off unless a
level sets `rgb_glass: optical` (draft level `sim/configs/V2_place_opt.yaml`); `sim/regression_test.py` gives 190 of
190 arrays identical for the default scene, `L0_vis`, `L0_vis_place` and `L0_vis_ell` with the final code.

*What was done, in order, with what departed from the listing.*
1. **The numbers from real images, as listed: could not be met.** Real colour pairs: 4079 from all frames of
   TRansPose. On the 139 windows with plain surroundings real glass is brighter than its surroundings (median of
   the glass minus the others +18 grey levels, 95th percentile +41); the model as listed darkens the glass's area
   (-10, +1 at the best grid point). Also the simulator's windows are mostly not plain (the coaster's ring, the
   gripper), which the listing had assumed.
2. **Revision 0, before any judge was looked at.** (a) The plainness limit is applied to the simulator's windows
   too. (b) The three numbers are replaced by quantities that do not depend on what else is in the window: the
   regression of the glass's median luminance on its surroundings' (2376 real windows: 0.68 x surroundings + 43,
   r 0.79) for the veil, and the spread inside the glass in excess of the spread around it for highlights (p95 -
   median: real +18.7) and edge (p5 - median: real -3.0). (c) Read as "the glass pulls its area 0.32 of the way to
   the average around it", the regression gave F0 = 0.30 with T0 = 1; with it k_s = 1.2 and w_rim = 0 matched the
   two spreads (19.9, -2.2).
3. **A bug, found by the floor.** In the first run of round 1 the colour-blind pairs differed (judge's AUC 0.58):
   the segmentation image carries stray single pixels with the ids of the parked glasses along edges, and the model
   drew them as glass. Fixed: only the glasses a scene names are drawn, and parts of a mask smaller than 6 pixels
   are dropped. The blind pairs are then the same image (1 pixel by 1 grey level in some, the renderer's own).
4. **Round 1 (judge A, one network as listed):** shows 0.53 against the 0.62 it needed, the window with the glass
   higher in 0.61 of the pairs; faint 0.47. Two trainings of the same judge on the same pairs differed by 0.07 on
   the alpha model, so from round 2 on a judge is the mean of five networks (seeds 0 to 4); a departure from the
   listing, made for both judges before judge B was run.
5. **Round 2 (judge A): which part moves the judge.** 1200 episodes per variant; the judge passes a state at 0.58.

   | "shows" under | judge A's AUC | window with the glass scored higher |
   |---|---|---|
   | the alpha model | 0.30 | 0.07 |
   | the veil alone (F0 0.30, no highlights) | 0.33 | 0.03 |
   | revision 0 (veil 0.30 and highlights) | 0.54 | 0.64 |
   | the edge line alone | 0.61 | 1.00 |
   | highlights alone | 0.68 | 0.99 |
   | highlights twice as strong | 0.69 | 0.99 |
   | highlights and edge line | 0.70 | 1.00 |
   | T0 0.92 and F0 0.08 as first listed, with highlights | 0.70 | 1.00 |

   A smooth veil over the glass's area is counted against glass, as the alpha model's tinted solid is; highlights
   and the edge line are counted for it.
6. **Round 3, the final model:** T0 0.92 and F0 0.08 as first listed (the veil of revision 0 is withdrawn: at the
   simulator's table, luminance 166, the same regression says the glass's area is 10 grey levels darker than its
   surroundings, and these constants give -9; that is beyond the real windows' range, whose surroundings end at
   137), and with them from real images k_s = 1.0 and w_rim = 0.1 (spreads 19.0 and -2.3 against 18.7 and -3.0).
   Judge A: shows 0.71 (needs 0.59), the window with the glass higher in 1.00 of the pairs; faint 0.56.

*The final read-out (`readout3.txt`), judge B once.*

| | optical model | alpha model |
|---|---|---|
| rule 1, floor: the simulator's own network on the colour-blind pairs | 0.50 | (0.50, section 11) |
| the simulator's own network: shows / faint | 1.00 / 0.92 | 1.00 / 1.00 |
| **rule 2, judge B on "shows"** (its own AUC 0.78, passes at 0.64) | **0.50, the glass higher in 0.42 of the pairs: no** | 0.33, in 0.03: no |
| judge B on "faint" | 0.50, in 0.35: no | 0.47, in 0.26: no |
| judge A on "shows", for comparison (the judge the model was developed with; its own AUC 0.67) | 0.71, in 1.00 | 0.33, in 0.10 |
| **rule 3, the simulator's network on all real pairs** (the real network's own 0.84, passes at 0.67) | **0.49: does not transfer** | 0.64: does not transfer |

- **Rule 2: no.** The judge that was kept aside does not respond to the optical model's glass at all; the judge
  the model was developed with responds in every pair. The two judges learned real glass from different halves of
  the sequences and do not agree on what the simulator's highlights are. This is what the second judge was for:
  three rounds were enough to fit one judge.
- **Both judges agree on one thing: the alpha model is counted against glass** (0.33 and 0.33, the window with the
  glass lower in nine pairs of ten). The optical model takes that away (0.50 for judge B) and no more.
- **Rule 3: no for both**, and the alpha model's network is the less far (0.64 against 0.49).
- **What the optical model does have:** its highlights, its edge and the darkening of its area are set to the
  statistics of real glass, and the alpha model's are not (it puts a tint over the area that real glass does not
  have). That is by construction and is not a test.

**What this check can and cannot decide.** The real colour images at hand are close-up views from above of vessels
on patterned cloth in one laboratory; the simulator's camera sees a small glass on a plain table from the side. A
network trained on one half of those sequences is at 0.67 or 0.78 on the other half, and two such networks disagree
about the simulator. For depth and thermal the check of section 11 had signals that carry over (a hole, a patch of
another temperature); for the colour image of glass it does not have the data to decide. That needs colour images of
glass from the robot's own camera, i.e. choice (c) of section 11.

**What follows, fixed by the listing:** the colour image of the second version stays the alpha model unless the user
decides otherwise, and the paper states the limit with these numbers. The optical model stays in the code as an
option that is off. Open for the user: keep alpha (the rule's default), or take the optical model without the
check's support (it is closer to real glass in its image statistics and to the eye, it is not counted against glass
by either judge, and "faint" is harder under it; whether "colour shows" is still enough for a policy at 96 x 96 would
have to be measured first).

## Files

`real_data/glass_visibility.py` (real sensors), `sim/measure_visibility.py` (the simulator, same functions),
`real_data/findability.py` (sections 11 and 12: the checks against real images), `real_check/` (their read-outs).
`sim/glass_optics.py` (section 12: the optical colour model of the glass, off by default), `sim/configs/V2_place_opt.yaml`.
`measurements/` (their outputs of 11/26). `sim/configs/V2_place.yaml`, `sim/configs/V2_place_warm.yaml` (draft levels).
T2: `sim/assets/gen3_lite/align_scene.xml`, `sim/configs/V2_align.yaml`, `sim/env.py` (`ALIGN_OBJECTS`, `_mat_target`,
`placed`), `sim/teacher.py` (`_plan_align`).
Unseen objects: `sim/assets/gen3_lite/place_unseen_scene.xml`, test set `u_cells11_unseen`. T3:
`sim/assets/gen3_lite/two_scene.xml`, `sim/configs/V2_two.yaml`, `sim/env.py` (`_second_glass`, `_two_targets`,
`on_coaster`), `sim/sensors.py` (the second glass's depth), `sim/teacher.py` (`_plan_two`). `sim/task_check.py`.
The angle of T2: `policy/hindsight_angle.py`, `policy/config/task/glass_pick_hcue2a.yaml`, `policy/glass_dataset.py` and
`policy/glass_gym_env.py` (the observation `hang`), `policy/train_cuegate.sh` (`ANGLE=1`).
Runs of the pilot and of step 4: one folder per run (`curve.json`, `train_logs.json.txt`, per test set `eval_log.json`
and `episodes.json`); read-outs `readout_pilot.txt`, `readout_thermal_rule.txt`, `readout_tasks.txt`
(`policy/readout_v2_pilot.py`, `policy/readout_v2_thermal_rule.py`, `policy/readout_v2_tasks.py`); the repeated pilot:
`sim/configs/V2b_place.yaml`, `policy/readout_v2_pilot_b.py`, `readout_pilot_b.txt`.
