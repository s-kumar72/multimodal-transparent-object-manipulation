# Visibility benchmark, second version: design draft for review

**Status (11/26): a draft. Nothing of it is implemented, collected, trained or frozen.** Decision of the user 11/26:
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
| thermal | shows | a glass at rest, usual contrast | 26 to 103 noise units (0.5 to 2.2 degC), either sign | 4 to 18 degC |
| thermal | faint | a glass at rest, low contrast | 3 to 10 noise units (0.06 to 0.21 degC) | about 1.7 degC |
| thermal | blind | glass and background in equilibrium | below 3 noise units | below about 5 |
| colour | shows | usual contrast against the table | alpha 0.5 (median contrast 27) | the same |
| colour | faint | low contrast | alpha 0.2 (median 13) | the same |
| colour | blind | contrast below 3 grey levels | alpha 0 to 0.04 | alpha 0.001 |

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

## Files

`real_data/glass_visibility.py` (real sensors), `sim/measure_visibility.py` (the simulator, same functions),
`measurements/` (their outputs of 11/26).
