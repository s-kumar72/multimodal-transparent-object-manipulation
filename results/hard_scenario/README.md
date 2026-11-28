# Transparent-object manipulation with colour, depth and thermal: where the work stands

An overview of the experiments under `results/hard_scenario/`, for readers who have not followed them day by day.
Every number below is taken from the README or read-out file named next to it; those files hold the full tables,
the run lists (written before the runs started) and the limits of each result. State of 2026-11-28.

## 1. The project

A robot arm (Kinova Gen3 lite, MuJoCo) picks up and places glass objects from three camera images of 96 x 96 pixels:
colour, depth and thermal. Glass is the case in which each of the three sensors fails in its own way: the colour
image may hardly show it, a depth sensor returns holes or the surface behind it, and a glass at room temperature
disappears from the thermal image. The policies are Diffusion Policy trained on demonstrations of a scripted teacher.

The paper offers two things:

- **a benchmark** in which every episode belongs to a *visibility cell* (which of the three sensors shows the glass,
  fully, faintly or not at all), so that a policy's use of each sensor can be read per cell;
- **a method**, hindsight cue gating, that learns from the demonstrations alone where the glass is and which sensors
  to trust.

One rule holds for every baseline and for the method, in training and at test time: **the input is sensor images and
joint values, and the supervision is the demonstrated actions.** Cells, visibility flags, glass masks and object
positions exist in the simulator and are used for the reports only. Runs that use more (for example a segmentation
pre-training on simulator masks) are reported in a category of their own ("extra supervision"), and policies that are
told the cell are called *references*, not baselines.

Run lists and read-outs are committed before the runs start; an analysis that was not fixed beforehand is labelled
"not fixed in advance" in the READMEs and here.

## 2. Status (2026-11-28)

| part | state |
|---|---|
| Benchmark, first version (`L0_vis`, `L0_vis_ell`) | frozen 11/22 with baselines and comparison rules ("Stage 6") |
| Method on the first version | three seeds on both levels; "better in nearly every case" holds for `full` and `cue` |
| Controls of the method | done with one seed, the decisive one (one network) with three; cue fits repeated five times |
| Baselines from the literature | three seeds each, finished 11/28; none is above the plain rgbdt baseline |
| Pick-and-place level `L0_vis_place` | baselines and method with seed 42; seeds 43 and 44 are in the queue |
| Benchmark, second version (`bench_v2`) | designed, checked against real sensor data, piloted and **frozen on 11/28** (levels, test sets, protocol, run list and read-out); the tuning runs and the full run are in the queue |
| Real robot | after the benchmark, the method and the simulation are finished |

## 3. Benchmark, first version

**Level `L0_vis`** ([`grid_bench/README.md`](grid_bench/README.md), `sim/configs/L0_vis.yaml`). Eleven cells: the
seven combinations of shows / blind over the three sensors, three cells in which one sensor shows the glass faintly
and the others are blind, and one with all three faint.

| sensor | shows | faint | blind |
|---|---|---|---|
| colour | glass at alpha 0.5 | alpha 0.2 | clear glass, not in the colour image (alpha 0.001) |
| depth | 40% of the glass pixels are holes | 20% | see-through, no holes |
| thermal | 5-35 degC from the room | 2.7-3.3 degC from the table behind it | within 0.1 degC of the table |

Training set: 700 demos (140 of cell 0, 70 each of cells 1 to 3, 50 each of cells 4 to 10). Test sets: `j_cells11`
(training conditions) and `k_cells11_heldout` (held-out backgrounds, lighting and distractors), 660 episodes each, 60
per cell, so 1320 episodes per run. Protocol: 101 epochs, the last checkpoint, default settings, nothing selected on
the test sets, seeds 42, 43 and 44. The headline number is the macro average over the eleven cells, always reported
with the table per cell.

**Level `L0_vis_ell`** ([`contact_check/README.md`](contact_check/README.md)). `L0_vis` seed for seed, with the
contact model of the place level (elliptic friction cones). It is used for evaluation only. Every checkpoint succeeds
more often on it than on `L0_vis`; the order of the policies and every verdict are the same on both. It is the level
of the paper's pick table.

**Plain baselines**, Diffusion Policy with ImageNet-initialised encoders, mean of three seeds (standard deviation):

| run | `L0_vis` | `L0_vis_ell` |
|---|---|---|
| rgbdt | 0.428 (0.013) | 0.587 (0.025) |
| rgbd | 0.397 (0.009) | 0.523 (0.010) |
| rgbt | 0.194 (0.004) | 0.304 (0.008) |
| rgb | 0.153 (0.011) | 0.237 (0.020) |
| best baseline of each cell, averaged | 0.458 | 0.624 |
| reference (best single-sensor policy of the cell; told the cell, one seed) | 0.674 | not evaluated |

Source: [`contact_check/README.md`](contact_check/README.md), "Results of round 2"; the per-cell tables on `L0_vis`
are in [`grid_bench/README.md`](grid_bench/README.md), "Stage 6".

What the baselines show: no baseline is the best everywhere; the plain policies lean on depth (on `L0_vis_ell` the
best baseline of a cell is at 0.78 to 0.91 where depth shows the glass and at 0.17 to 0.54 where it does not); and on
`L0_vis` rgbdt is below a policy with fewer sensors in cells where the extra sensors do not show the glass.

**Comparison rules for a method (Stage 6, fixed 11/21-11/22):** (1) the same information as the baselines; (2)
default settings, nothing selected on the test sets; (3) three seeds, two policies compared on the same episodes
(sign test); (4) per cell against the best baseline of that cell; (5) "better in nearly every case" = no cell in
which the method is below that best baseline with p < 0.05, and a macro average above every baseline.

## 4. The method: hindsight cue gating

[`cue_gate/README.md`](cue_gate/README.md).

- **Cue.** A demonstration says in hindsight where the glass was: the tool position at the first step that closes
  the gripper. Per sensor, a small fully convolutional network learns from the first frames of every demo a map over
  the image of where that grasp will be.
- **Gate.** Whether a sensor's peak can be trusted is learned without labels by cross-fitting: the map of a demo
  comes from a network that was not trained on it, and "the peak was within 2 pixels of the hindsight grasp pixel" is
  the label for a small verifier that looks at the image patch around the peak. Its output is the reliability of the
  sensor in this episode.
- **Policy.** Diffusion Policy with the grasp point of the trusted sensors, its confidence and the three
  reliabilities in the conditioning, computed once from the first frame. In the arm `full` the image feature of an
  untrusted sensor is also replaced by a learned embedding; the arm `cue` leaves the image features as they are.

Information used: sensor images, joint values, demo actions; arm kinematics, the camera calibration and ImageNet
weights, which the rules allow to every run.

**Main results** (macro average, mean of three seeds with standard deviation unless noted):

| policy | `L0_vis` | `L0_vis_ell` |
|---|---|---|
| plain rgbdt baseline | 0.428 (0.013) | 0.587 (0.025) |
| `full` | 0.792 (0.004) | 0.904 (0.007) |
| `cue` | 0.797 (0.004) | 0.915 (0.002) |
| `blind` (cue and joints, no images; seed 42) | 0.783 | 0.888 |
| `gate` (gated image features, no grasp point; seed 42) | 0.507 | |

Sources: [`cue_gate/README.md`](cue_gate/README.md) ("Results of round 1", "Results of the seeds", "Results of round
4"), [`contact_check/README.md`](contact_check/README.md) (rounds 2 and 3).

Per cell on `L0_vis_ell`, `full` with three seeds pooled against the best plain baseline of the cell
([`contact_check/readout_seeds.txt`](contact_check/readout_seeds.txt)):

| cell | sensors that show the glass | best baseline | `full` | |
|---|---|---|---|---|
| 0 | rgb+depth+thermal | 0.911 (rgbdt) | 0.983 | above |
| 1 | depth+thermal | 0.908 (rgbdt) | 0.953 | above |
| 2 | rgb+thermal | 0.542 (rgbt) | 0.964 | above |
| 3 | rgb+depth | 0.883 (rgbdt) | 0.969 | above |
| 4 | thermal | 0.236 (rgbt) | 0.839 | above |
| 5 | depth | 0.903 (rgbdt) | 0.889 | not separated |
| 6 | rgb | 0.414 (rgb) | 0.958 | above |
| 7 | rgb (faint) | 0.314 (rgb) | 0.958 | above |
| 8 | depth (faint) | 0.783 (rgbdt) | 0.808 | not separated |
| 9 | thermal (faint) | 0.172 (rgbt) | 0.689 | above |
| 10 | all three (faint) | 0.794 (rgbdt) | 0.928 | above |

- **"Better in nearly every case" holds** for `full` and for `cue` on both levels: no cell below its best baseline,
  and a macro average above every baseline. The cells in which the method is not ahead are those in which depth
  alone shows the glass.
- **Visibility combinations that are not in the training set** (15 new combinations, `L0_vis`): `cue` 0.849 against
  0.574 for the plain rgbdt baseline with seed 42 ([`cue_gate/readout_round3_new_cells.txt`](cue_gate/readout_round3_new_cells.txt)).
- **Pick-and-place level `L0_vis_place`** (seed 42, [`place_bench/README.md`](place_bench/README.md)): plain
  baselines rgbdt 0.536, rgbd 0.535, rgbt 0.264, rgb 0.096; the method with a grasp cue and a place cue 0.855 (arm
  `full`) and 0.867 (arm `cue`). The arm `full` is above the best baseline in six cells, not separated in four and
  **below it in one** (cell 5, depth alone: 0.867 against 0.942, p = 0.049). One seed; seeds 43 and 44 are in the
  queue.

## 5. What the controls say, and what is not claimed

Seed 42, the same episodes; the arm `cue` with one part changed
([`cue_gate/README.md`](cue_gate/README.md), "Round 2 as a whole"; [`contact_check/README.md`](contact_check/README.md),
"Results of round 3, first part"):

| run | what differs from the method's cue | `L0_vis` | `L0_vis_ell` |
|---|---|---|---|
| `cue` | | 0.795 | 0.914 |
| one network | one network on all sensors forms the cue, no gate | 0.730 | 0.911 |
| gate on map numbers | the gate does not look at the image | 0.695 | 0.825 |
| no gate | the three maps are averaged | 0.502 | 0.655 |
| auxiliary loss | no cue; the same hindsight label as an extra loss on the plain policy | 0.226 | 0.439 |
| plain rgbdt | no cue, no label | 0.414 | 0.560 |

`cue` against the one-network control with three seeds
([`cue_gate/readout_round4.txt`](cue_gate/readout_round4.txt), [`contact_check/readout_round4_ell.txt`](contact_check/readout_round4_ell.txt)):

| setting | level | `cue` | one network | verdict fixed before the runs |
|---|---|---|---|---|
| training cells | `L0_vis` | 0.797 | 0.722 | holds over the seeds |
| training cells | `L0_vis_ell` | 0.915 | 0.909 | does not hold (p = 0.24) |
| 15 new combinations | `L0_vis` | 0.854 | 0.779 | holds over the seeds |
| 15 new combinations | `L0_vis_ell` | 0.961 | 0.945 | holds over the seeds (11/28) |
| trained without cells 4 to 9, the cells left out | `L0_vis` | 0.675 | 0.554 | holds over the seeds |
| trained without cells 4 to 9, the cells left out | `L0_vis_ell` | 0.829 | 0.740 | holds over the seeds |

The cue networks were then fitted five times, with no policy involved
([`cue_gate/README.md`](cue_gate/README.md), "Results of 6a"). The fit the policies were trained with is the one fit
of five in which the method's cue is the more accurate on the training cells; over the five fits the one network's cue
is right more often there (0.914 against 0.888 within 1.5 cm) and on the new combinations (0.955 against 0.941). In
the cells left out of training the method's cue is ahead in every fit (0.785 against 0.643).

What follows from this:

- **Claimed:** a grasp point learned from the demos' own hindsight, given to the policy as conditioning, takes the
  plain policy from 0.59 to 0.92 on `L0_vis_ell` with no cell below its best baseline. A gate among the sensors is
  needed to form that point (0.914 against 0.655 without one), it has to look at the image (0.825 for a gate on map
  numbers), and the same label as an auxiliary loss does not do it.
- **Not claimed:** that maps per sensor with an explicit gate are better than one network over all sensors on the
  conditions it was trained on. On the level of the paper's table the two are level there, and the differences on
  `L0_vis` and on the new combinations (1.6 points on `L0_vis_ell`) rest on one fit of the cue networks.
- **Where the explicit gate is consistently ahead:** when two sensors are blind at the same time for the first time
  (the cells left out of training), in every cue fit and in the policies on both levels.
- Gating the image features adds nothing once the cue is in the conditioning (`full` is not above `cue`, and below
  it on the new combinations of `L0_vis_ell`: 0.948 against 0.965).
- Still running: two more training sets with cells left out (round 5), the
  success against the number of demos (6b), the true grasp point as an upper reference (6c).

## 6. Baselines from the literature

[`lit_base/README.md`](lit_base/README.md), [`lit_base/readout_seeds.txt`](lit_base/readout_seeds.txt). Implemented
in the benchmark's Diffusion Policy with the papers' settings or stated defaults; none was tuned on the benchmark.
Three seeds each, finished 11/28.

| run | `L0_vis`, mean of three seeds | `L0_vis_ell` | 15 new combinations |
|---|---|---|---|
| plain rgbdt | 0.428 | 0.587 | |
| self-attention fusion of the sensors' tokens (MulSA) | 0.432 | 0.578 | 0.574 |
| sensor dropout in training | 0.423 | 0.573 | 0.539 |
| stage-guided dynamic fusion (MS-Bot) | 0.156 | 0.244 | 0.205 |
| `cue` | 0.797 | 0.915 | 0.854 |

- Self-attention fusion and sensor dropout are not separated from plain rgbdt on either level (p from 0.095 to 0.6).
  Sensor dropout moves success from the cells with depth to those without and the sum stays where it was.
- With both among the baselines, the best baseline of a cell is one of them in some cells (macro average of the best
  baseline per cell 0.470 on `L0_vis` and 0.639 on `L0_vis_ell`), and the method's Stage 6 verdict still holds on
  both levels.
- **MS-Bot comes with a caveat.** Its three seeds are 0.161, 0.257 and 0.051 on `L0_vis`, and in every seed the
  training loss stays on a plateau for a third to three quarters of the training (until epoch 37, 74 and 39 of 101).
  The implementation follows the paper's parts and the benchmark's protocol and was not tuned; its number should not
  be read as what the published method can reach.

## 7. Learned attention weighting (extra supervision)

[`fusion_vis/README.md`](fusion_vis/README.md). The project's second fusion method (`fusion/attention/`): a glass
segmentation with learned per-sensor weights as the image encoder of the policy. Its pre-training uses the
simulator's glass masks, so it is reported as a method with extra supervision, next to the comparison of Stage 6 and
not inside it.

- The variant with a full-resolution position readout and keypoints (P9) reaches 0.532 on `L0_vis` with seed 42 and
  0.526 over three seeds (0.458 to 0.588), above the plain rgbdt baseline (0.428). Against the best baseline of each
  cell it is above in cells 2, 4, 6, 7, 9 and 10 and below in cells 0, 1, 3 and 5, where depth shows the glass fully.
- Without the pre-training it is at 0.022, with a frozen encoder at 0.137, without the thermal image at 0.260.
- Six further variants of the ablation list were run with seed 42 (round 3): none is above P9.
- On the same episodes the method of section 4 is above P9 (0.792 against 0.526 over three seeds).

## 8. Benchmark, second version (`bench_v2`)

[`bench_v2/README.md`](bench_v2/README.md). Frozen on 11/28 (section 13 of that README). The numbers in this section
are the pilots'; by their listing none of them enters a table of the benchmark, and every run of the tables is
trained again.

**Why.** Three objections to the first version that cannot be mended by adding to it: the sensor failures are numbers
we picked; demos and evaluation use different contact models; and the one task is solved by one point (a policy with
the cue and the joints and no image is at 0.888 on `L0_vis_ell`). The first version stays as the development record
and as the fallback.

**What changes.**

- *Sensor states from measurements on real glass* (five public datasets, `real_data/glass_visibility.py`): depth
  "shows" is a hole-type sensor (0.80 to 1.00 of the glass pixels missing), "faint" 0.10 to 0.35, "blind" a
  see-through sensor; thermal contrast is set in units of the camera noise from tabletop glass at rest (real p10 /
  p25 / median / p75 / p90: 10 / 26 / 47 / 67 / 103 units, about 0.2 to 2.2 degC), ten times weaker than the "shows"
  of `L0_vis`.
- *One contact model* for demos and evaluation.
- *Three tasks:* T1 `place` (pick the glass and put it on a coaster), T2 `align` (an elongated glass lying at a
  random angle, grasped across its narrow side and put down on a mat in the mat's direction), T3 `two` (two glasses
  with separate visibility states, one on each of two coasters).

**Checks against real sensor images** (sections 11 and 12 of that README). A network trained on real glass was shown
the simulator's glass, and the reverse. Depth "shows" (a hole) carries over both ways. Thermal "shows" is supported
one way (the real-glass network responds to the simulator's glass at 30 to 103 noise units). The colour image is not
supported: the renderer blends a tinted solid and does not refract or reflect. An optical model with highlights and
an edge line was built and did not pass a judge network that had been kept aside, so the colour image stays the alpha
model and the limit is stated.

**The thermal rule and its verdict.** Before any pilot run it was fixed that a thermal state is changed only if a
cell cannot be solved, and how. On the first states (`V2_place`: faint 3 to 10 units, shows 10 to 103) cell 9
(thermal faint alone) was below 0.30 in all four pilot runs (best 0.158) and in the reference run of the rule
(0.108; that reference was below 0.30 in every cell, so by itself it is weak evidence). By the rule the states became
`V2b_place` (faint 10 to 26 units, the real p10 to p25; shows 26 to 103, the real p25 to p90) and the four runs were
repeated once.

**Pilot of T1 on `V2b_place`** (seed 42, 1320 episodes per run;
[`bench_v2/readout_pilot_b.txt`](bench_v2/readout_pilot_b.txt)):

| cell | sensors that show the glass | plain rgbdt | plain rgbd | `cue` | one network |
|---|---|---|---|---|---|
| 0 | rgb+depth+thermal | 0.975 | 0.983 | 0.983 | 0.992 |
| 1 | depth+thermal | 1.000 | 0.975 | 0.958 | 0.992 |
| 2 | rgb+thermal | 0.175 | 0.092 | 0.925 | 0.833 |
| 3 | rgb+depth | 1.000 | 0.983 | 0.983 | 0.983 |
| 4 | thermal | 0.083 | 0.025 | 0.683 | 0.342 |
| 5 | depth | 0.992 | 0.975 | 0.967 | 0.983 |
| 6 | rgb | 0.108 | 0.067 | 0.858 | 0.858 |
| 7 | rgb (faint) | 0.075 | 0.067 | 0.850 | 0.767 |
| 8 | depth (faint) | 0.625 | 0.583 | 0.633 | 0.675 |
| 9 | thermal (faint) | 0.142 | 0.067 | 0.217 | 0.092 |
| 10 | all three (faint) | 0.575 | 0.642 | 0.908 | 0.908 |
| macro average | | 0.523 | 0.496 | 0.815 | 0.766 |
| the same on `V2_place` | | 0.514 | 0.505 | 0.784 | 0.739 |

- The plain baselines depend on depth alone: 0.975 to 1.000 where depth shows the glass fully, 0.025 to 0.175 where
  it does not show it at all.
- `cue` against plain rgbdt on the same episodes: only `cue` succeeds in 443, only the baseline in 57 (p = 4.3e-75);
  against the one network 141 and 76 (p = 1.2e-05).
- **Cell 9 is again below 0.30 in all four runs**, so the stop that was fixed beforehand applied and the decision
  was taken outside the rule (decisions of 11/28, below). Not fixed in advance: in cell 9 the method's cue is within 1.5 cm of the glass in 0.21
  of the episodes; success is 0.68 where it is and 0.09 where it is not. What is missing is locating the glass in the
  faint thermal image, not acting on it.

**Pilots of T2 and T3** (seed 42, on the first thermal states;
[`bench_v2/readout_tasks.txt`](bench_v2/readout_tasks.txt)): T2 plain rgbdt 0.564, `cue` with the hindsight grasp
angle 0.828 (higher in each of the eleven cells). T3 plain rgbdt 0.143 (one glass on a coaster in 0.442 of the
episodes, both in 0.143). The method has not been run on T3: its cue for two targets is not written yet.

**Decisions of 11/28.**

1. Cell 9 stays and is marked as *open*: the tables carry all eleven cells and the macro average over eleven, with
   the statement that no policy of the pilot reaches 0.30 in it.
2. The thermal states are those of `V2b_place`, for all three tasks (draft levels `V2b_align`, `V2b_two`).
3. T2 is trained on 630 demos (its three glass shapes equally often), T3 on 700.
4. A reference per task that gets no glass in any image is added, so that the share of the layout (below) is a
   number in the tables.
5. The full run of the second version goes before the remaining runs of the first version.

**What is frozen** (section 13 of [`bench_v2/README.md`](bench_v2/README.md)).

| | T1 place | T2 align | T3 two |
|---|---|---|---|
| level | `V2b_place` | `V2b_align` | `V2b_two` |
| demos | 700 | 630 (210 per glass shape) | 700 |
| headline test sets | `j_cells11`, `k_cells11_heldout`, 660 episodes each | the same | `j_pairs12`, `k_pairs12_heldout`, 360 each |
| generalisation sets | 15 unseen combinations of states; unseen glass shapes | the same | none |
| policies, three seeds | four plain baselines, `cue`, the one network; on T1 also the baselines from the literature, the second policy, the true grasp point and the left-out cells | four plain baselines, `cue` with the grasp angle, the one network | four plain baselines; the method once its cue for two targets is written |

- A tuning budget is part of the protocol: on T1 with seed 42, plain rgbdt, `cue` and the one network are each
  trained with {101, 201} epochs x {1e-4, 3e-4} learning rate and scored on 660 validation episodes whose seeds are
  in no test set; the best setting of each is its setting for every task and seed.
- A layout reference per task (plain rgbdt trained and tested on the task's scenes with the glass in no image)
  measures what the layout alone gives.
- The read-out (`policy/readout_v2_full.py`) and its rules are committed with the freeze: per task, `cue` against the
  best baseline of every cell over three seeds, with the same "better in nearly every case" rule as Stage 6, and
  `cue` against the one network. Cell 9 counts as every other cell and is marked as open on T1 (on T2 the pilot's `cue` run was at
  0.417 in it).

**Known limits, to be stated in the paper.**

- The colour image of the glass is an alpha model; "colour shows" is closest to a transparent object whose body is
  visible (frosted or tinted glass), not to clear glass with highlights.
- *The layout reveals the glass.* Distractors and the place target keep a clearance from the glass, and the glass
  spawns in a small region, so the positions of the other things alone put the glass within a median 3 to 4 cm (3.1
  cm on T2, 3.8 cm on T1). It is open to every policy alike and is not simulator information, but a cell in which a
  sensor is "blind" is not a cell in which that image says nothing about where to grasp
  ([`bench_v2/blind_layout.txt`](bench_v2/blind_layout.txt)).
- With the thermal states of `V2b_place`, real contrasts of 3 to 10 noise units (6% of the measured regions) are in
  no state.
- All numbers of the second version so far are one seed, and by their listing the pilots decide the design only:
  nothing of them enters the tables of the frozen benchmark.

## 9. What is running and what comes next

- **Remaining runs of the first version** (in the queue): two more
  training sets with cells left out (`nopair`, `nofaint`); seeds 43 and 44 on the place level; a second policy (ACT
  action head with and without the cue, [`second_policy/README.md`](second_policy/README.md)); the true grasp point
  in the cue as a reference; success against the number of demos (350 and 140).
- **Second version:** frozen on 11/28. The twelve tuning runs go first, then everything once: about 980 GPU hours
  with the benchmark's setting, about eleven days on four GPUs. Before the method runs on T3, its cue for two
  targets has to be written.
- **Real robot:** after the benchmark, the method and the simulation results are finished.
- If the second version is late, the paper is written on the first version.

## 10. Index of `results/hard_scenario/`

| folder | what it holds | read-out |
|---|---|---|
| [`grid_bench/`](grid_bench/README.md) | design of the visibility cells, `L0_vis`, the baseline programme, Stage 6 (frozen benchmark and rules), failure analysis of the baselines | `policy/summarize_cells.py` |
| [`contact_check/`](contact_check/README.md) | `L0_vis_ell`; three-seed tables of baselines and method on it; the method's controls on it | `policy/readout_contacts.py`, `readout_contacts_seeds.py`, `readout_contacts_round3.py` |
| [`cue_gate/`](cue_gate/README.md) | the method: perception, rounds 1 to 6 (arms, controls, new combinations, left-out cells, seeds, cue fits) | `policy/readout_cue_gate.py`, `readout_cue_seeds.py`, `readout_round3.py`, `readout_round4.py`, `readout_cue_fitseeds.py` |
| [`lit_base/`](lit_base/README.md) | baselines from the literature (MulSA, sensor dropout, MS-Bot), what is the papers' and what is not | `policy/readout_lit_base.py`, `readout_lit_seeds.py` |
| [`place_bench/`](place_bench/README.md) | pick-and-place level `L0_vis_place`: baselines, the method with a place cue | `policy/readout_place.py`, `readout_place_method.py`, `readout_place_seeds.py` |
| [`second_policy/`](second_policy/README.md) | the cue with an ACT action head (runs listed, not finished) | `policy/readout_second_policy.py` |
| [`fusion_vis/`](fusion_vis/README.md) | learned attention weighting on `L0_vis` (extra supervision), three rounds | `fusion/attention/readout_l0_vis.py`, `readout_l0_vis_round2.py`, `readout_l0_vis_round3.py` |
| [`bench_v2/`](bench_v2/README.md) | second version: measurements on real glass, states, tasks, pilots, the thermal rule, checks against real images, the optical colour model, the frozen benchmark (section 13) | `policy/readout_v2_pilot.py`, `readout_v2_pilot_b.py`, `readout_v2_thermal_rule.py`, `readout_v2_tasks.py`; for the frozen benchmark `readout_v2_tune.py`, `readout_v2_full.py` |
| [`fusion_bench/`](fusion_bench/README.md) | the two fusion methods on the earlier level `L0_bench` and the gating rounds that followed; why that level was left | |
| [`benchmark/`](benchmark/README.md), [`build_up/`](build_up/README.md), [`easy_l1/`](easy_l1/README.md), [`thermal_value/`](thermal_value/README.md), [`level_sweep_500/`](level_sweep_500/README.md), [`diagnosis/`](diagnosis/README.md), [`sweep/`](sweep/README.md) | the search for a level before the visibility cells: hard levels L0 to L4b, what thermal adds on top of depth, scene factors one at a time | |

Each run has a folder under its group with the training curve, the evaluation log and the per-episode records of
every test set; the read-outs are computed from those records and need no new run.

## 11. Launching a run

Runs are started from the repo root with `scripts/run_experiment.sh <level> <inputs> <demos>`, which collects the
demos if they are missing, trains, and evaluates the last checkpoint; the knobs are environment variables described
at the top of the script. It expects the Diffusion Policy repo next to this one (`../diffusion_policy`) and a Python
3.11 environment with PyTorch, the simulator's requirements and Diffusion Policy's (`scripts/bridges2/setup.sh`
builds one). Rendering is headless (`MUJOCO_GL=egl`, set by the script).

The plain rgbdt baseline of the T2 pilot, as listed in [`bench_v2/README.md`](bench_v2/README.md):

```bash
GROUP=bench_v2 TOPK=0 NTEST=66 NEVAL=660 TEST_SETS="j_cells11 k_cells11_heldout" TAG=imagenet \
EXTRA="training.seed=42 task.env_runner.test_set=j_cells11 task.env_runner.max_steps=240 policy.obs_encoder.rgb_model.weights=IMAGENET1K_V1" \
bash scripts/run_experiment.sh V2_align rgbdt 700
```

Results land in `results/hard_scenario/<GROUP>/<run>/`. A table per cell of one or more runs, with paired
comparisons: `python policy/summarize_cells.py <run folders> [--versus <run folders>]`. Demo sets
(`results/data/*.zarr`) and checkpoints (`data/runs/`) are not in the repository.
