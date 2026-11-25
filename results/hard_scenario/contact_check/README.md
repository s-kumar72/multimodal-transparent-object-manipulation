# Contact check: how much of the pick's failures on `L0_vis` is the contact model?

Started 11/25 (approved by the user 11/25, step 1 of five). Evaluation only; nothing is trained and no level is
changed.

## Why

The baselines on the pick-and-place level pick the glass more often than the same inputs succeed on the pick level
`L0_vis`, on the same scenes and seeds ([`../place_bench/README.md`](../place_bench/README.md), results of step 2,
item 4): rgbdt 0.605 against 0.414. The first close of the gripper is as near to the glass on both levels; what
differs is that a near first close ends in a pick in 0.98 of the episodes there and in 0.75 here. The two levels
differ in the contact model (`L0_vis`: MuJoCo's default, pyramidal friction cones with impratio 1; `L0_vis_place`:
elliptic cones with impratio 10, set because a held glass crept down between the fingers while it was carried), but
the policies differ too (twice the gradient steps, longer demos), so the files could not separate the two.

This check separates them: the same checkpoints, the same scenes, only the contact model changed.

## The level `L0_vis_ell`

`sim/configs/L0_vis_ell.yaml` (`base: L0_vis.yaml`, `contacts: elliptic`, meta/ep_level id 37; in
`sim/configs/manifest.json`). `contacts` is a new field of the scene config (`sim/config.py`, default `default`: the
scene file's own model); `elliptic` sets elliptic friction cones with impratio 10 after the scene is loaded
(`sim/env.py`), the two options `place_scene.xml` sets. Everything else is `L0_vis`, seed for seed; the test sets are
`j_cells11` and `k_cells11_heldout` with the new level. No demos are collected on it.

Checks before any evaluation (11/25):

- **Existing levels are unchanged.** `python sim/regression_test.py --check` against arrays saved with the code before
  the change, for the default scene, `L0_vis` and `L0_vis_place`: 190 of 190 arrays identical in each.
- **`L0_vis_ell` against `L0_vis`, first frames of 20 seeds:** joints and thermal identical; the colour image differs
  by at most 3 grey levels in at most 15 of 27648 values, depth by 1 unit in at most 1 of 9216 pixels (the scene
  settles for a few steps at reset, and EGL's blending is not reproducible to the last level). The policies see the
  same first frame.
- **The teacher on `L0_vis_ell`** ([`teacher_check.txt`](teacher_check.txt)): 700 of 700 training seeds and 660 of
  660 on each test set, every object; no distractor moved.

## Runs (listed 11/25 01:27 KST, before any of them starts)

The last checkpoint of six finished runs of seed 42, each evaluated on `L0_vis_ell` with the final evaluation of the
benchmark (`j_cells11` and `k_cells11_heldout`, 660 episodes each, 120 per cell): `GPU=<g> bash
policy/eval_contacts.sh <run>`, results in `results/hard_scenario/contact_check/<run>/`. About 35 minutes per run.

| run | what it is | success on `L0_vis` (in git) |
|---|---|---|
| `L0_vis_rgbdt_700demos_imagenet` | frozen baseline, colour + depth + thermal | 0.414 |
| `L0_vis_rgbd_700demos_imagenet` | frozen baseline, colour + depth | 0.404 |
| `L0_vis_rgbt_700demos_imagenet` | frozen baseline, colour + thermal | 0.195 |
| `L0_vis_rgb_700demos_imagenet` | frozen baseline, colour | 0.158 |
| `L0_vis_rgbdt_700demos_hcg_cue` | hindsight cue gating, arm `cue` | 0.795 |
| `L0_vis_rgbdt_700demos_hcg_full` | hindsight cue gating, arm `full` | 0.789 |

These policies were trained on demos collected with the default contacts. The check changes the physics at test time
only; a level that used the other contact model from the start would also collect its demos with it.

## Read-out, fixed now (`python policy/readout_contacts.py > readout.txt`)

1. Per run: success on `L0_vis_ell` against the success of the same checkpoint on `L0_vis` (the final evaluation in
   git), on the same episodes (sign test, p < 0.05): over all episodes, in the cells in which depth shows the glass at
   least faintly (0, 1, 3, 5, 8, 10) and in the others.
2. The first close of the gripper under both contact models (and on the place level for the baselines): the share of
   episodes with the first close within 1.5 cm of the glass, the success after such a close and after one 0 to 1, 1 to
   1.5, 1.5 to 3 and over 3 cm away, and the closes per episode.
3. The question of this check, for the two baselines with depth (rgbdt, rgbd). With a = the success after a first
   close within 1.5 cm on `L0_vis` (0.748, 0.745), b = the share picked after one on `L0_vis_place` (0.983, 0.986) and
   x = the success after one on `L0_vis_ell`: the difference is **explained by the contacts** if x covers at least
   0.8 of the way from a to b, **not by the contacts** if it covers 0.2 or less, and **in part** in between.
4. The method against the rgbdt baseline on `L0_vis_ell`, same episodes (sign test), and the difference in success
   next to the one on `L0_vis` (+0.381 for `cue`, +0.375 for `full`).
5. Success per cell on `L0_vis_ell` next to `L0_vis`, for all six.

Not decided by this check: which contact model the pick benchmark uses. `L0_vis` and every result on it stay as they
are; the numbers of this check are reported next to them, whatever they are.

## Results (11/25 02:45 KST; seed 42, 1320 episodes per run, 120 per cell)

The six evaluations ran from 01:27 to 02:36 KST (queue of the fusion-vis worktree); none failed. Read-out:
[`readout.txt`](readout.txt).

**1. Success under the two contact models, the same checkpoint on the same episodes.**

| run | `L0_vis` | `L0_vis_ell` | only `L0_vis` / only `L0_vis_ell` | p | depth cells (0, 1, 3, 5, 8, 10) | the other cells |
|---|---|---|---|---|---|---|
| rgbdt baseline | 0.414 | 0.560 | 33 / 226 | 1.5e-36 | 0.639 -> 0.840 | 0.143 -> 0.223 |
| rgbd baseline | 0.404 | 0.514 | 34 / 180 | 3.4e-25 | 0.621 -> 0.783 | 0.143 -> 0.192 |
| rgbt baseline | 0.195 | 0.307 | 39 / 186 | 3.8e-24 | 0.185 -> 0.294 | 0.208 -> 0.322 |
| rgb baseline | 0.158 | 0.260 | 30 / 165 | 8.7e-24 | 0.144 -> 0.242 | 0.173 -> 0.282 |
| `hcg_cue` | 0.795 | 0.914 | 10 / 167 | 7.1e-38 | 0.836 -> 0.933 | 0.747 -> 0.892 |
| `hcg_full` | 0.789 | 0.898 | 19 / 163 | 1.0e-29 | 0.810 -> 0.918 | 0.763 -> 0.873 |

Every run succeeds more often with the contact model of the place level, by 0.10 to 0.15, in both groups of cells
(p < 0.001 in every row of the read-out). Nothing was retrained.

**2. The first close of the gripper.**

| run | level | first close within 1.5 cm of the glass | success after it | after a first close 0 to 1 cm away | 1 to 1.5 cm | 1.5 to 3 cm | closes per episode |
|---|---|---|---|---|---|---|---|
| rgbdt | `L0_vis` | 0.465 | 0.748 | 0.873 | 0.480 | 0.293 | 1.97 |
| rgbdt | `L0_vis_ell` | 0.461 | 0.929 | 0.974 | 0.834 | 0.590 | 1.77 |
| rgbdt | `L0_vis_place` (picked) | 0.479 | 0.983 | 0.986 | 0.973 | 0.685 | 1.19 |
| rgbd | `L0_vis` | 0.469 | 0.745 | 0.878 | 0.451 | 0.223 | 1.55 |
| rgbd | `L0_vis_ell` | 0.461 | 0.893 | 0.958 | 0.746 | 0.464 | 1.48 |
| rgbd | `L0_vis_place` (picked) | 0.499 | 0.986 | 0.996 | 0.959 | 0.610 | 1.14 |
| `hcg_cue` | `L0_vis` | 0.875 | 0.893 | 0.952 | 0.460 | 0.180 | 1.09 |
| `hcg_cue` | `L0_vis_ell` | 0.870 | 0.996 | 0.998 | 0.978 | 0.618 | 1.11 |
| `hcg_full` | `L0_vis` | 0.870 | 0.886 | 0.950 | 0.434 | 0.192 | 1.10 |
| `hcg_full` | `L0_vis_ell` | 0.870 | 0.979 | 0.991 | 0.889 | 0.583 | 1.16 |

Where the gripper first closes does not change with the contact model (as expected: the policy and the scene are the
same). What changes is whether a grasp near the glass holds: a first close 1 to 1.5 cm away ends in a success in 0.43
to 0.48 of the episodes with the default contacts and in 0.75 to 0.98 with the others; one 1.5 to 3 cm away in 0.18 to
0.29 against 0.46 to 0.62.

**3. Do the contacts explain why the glass is picked more often on the place level?** By the rule fixed above:

| baseline | a: after a near first close on `L0_vis` | b: on `L0_vis_place` | x: on `L0_vis_ell` | (x - a) / (b - a) | verdict |
|---|---|---|---|---|---|
| rgbdt | 0.748 | 0.983 | 0.929 | 0.77 | in part |
| rgbd | 0.745 | 0.986 | 0.893 | 0.61 | in part |

**In part**: the contact model covers about three quarters (rgbdt) and three fifths (rgbd) of the difference, below
the 0.8 that "explained by the contacts" was set at. The rest stays with what else differs between the two levels'
runs (the policies of the place level had twice the gradient steps and longer demos, and were trained on demos
collected with these contacts). Over all episodes: 0.560 on `L0_vis_ell` against 0.605 picked on the place level
for rgbdt, 0.514 against 0.574 for rgbd.

**4. The method against the rgbdt baseline.**

| method run | level | method | rgbdt baseline | difference | only the method / only the baseline | p |
|---|---|---|---|---|---|---|
| `hcg_cue` | `L0_vis` | 0.795 | 0.414 | +0.382 | 568 / 64 | 6.5e-102 |
| `hcg_cue` | `L0_vis_ell` | 0.914 | 0.560 | +0.355 | 509 / 41 | 8.7e-104 |
| `hcg_full` | `L0_vis` | 0.789 | 0.414 | +0.375 | 567 / 72 | 2.5e-96 |
| `hcg_full` | `L0_vis_ell` | 0.898 | 0.560 | +0.338 | 503 / 57 | 3.4e-90 |

The difference between the method and the baseline is nearly the same under both contact models (+0.34 to +0.36
against +0.38).

**5. Per cell**: in [`readout.txt`](readout.txt). Every cell of every run is at or above its value on `L0_vis`
except rgbd in cell 9 (0.083 against 0.092). The method's weak cells rise the most: `hcg_cue` in cell 8 from 0.625
to 0.875 and in cell 9 from 0.467 to 0.717.

**What the check says.**

1. With MuJoCo's default contacts a grasp that is at the glass often does not hold on `L0_vis`: every policy
   succeeds in 0.10 to 0.15 more of the episodes when only the contact model is changed. That part of the failures
   is a property of the simulated contact, not of what the sensors show.
2. The earlier reading that a pick needs the gripper within about 1 cm of the glass
   ([`../grid_bench/README.md`](../grid_bench/README.md), failure analysis) holds for the default contacts only. With
   the others a first close 1 to 1.5 cm away succeeds in 0.75 to 0.98.
3. The comparisons made so far do not turn over: the method is above the baseline by the same margin, and the order
   of the baselines is the same (rgbdt, rgbd, rgbt, rgb).
4. Under the contacts of the place level the method's failures are almost only episodes in which the gripper does
   not close near the glass: `hcg_cue` succeeds in 0.996 of the episodes with a first close within 1.5 cm (0.870 of
   all), so what is left to gain there is in the cue.
5. Limits: one seed; the policies were trained on demos collected with the default contacts; six checkpoints.

**Not decided here** (the user's decision): which contact model the pick level of the benchmark is reported with.
`L0_vis` and its results are unchanged. Until it is decided, the runs of round 4 of the method and of `../lit_base/`
are evaluated on `L0_vis` as listed, and their checkpoints are kept so that they can be evaluated on `L0_vis_ell` too.

## Round 2: the pick level of the benchmark with the contact model of the place level (decision of the user 11/25; listed 11/25 09:07 KST, before any of the runs starts)

**Decision of the user (11/25).** The pick level is reported with the contact model of the place level, from
evaluations of the existing checkpoints on `L0_vis_ell`, next to the numbers on `L0_vis`. Nothing is retrained and
no demos are collected again; `L0_vis` and every result on it stay as they are.

**Runs.** The last checkpoint of ten finished runs, each evaluated on `L0_vis_ell` as in the first round
(`j_cells11` and `k_cells11_heldout`, 660 episodes each; `GPU=<g> bash policy/eval_contacts.sh <run>`, about 35
minutes per run). With the six of the first round they complete the three seeds of the four baselines of the
comparison and of the method's arm `full`:

| runs | what they are |
|---|---|
| `L0_vis_rgbdt_700demos_imagenet_seed43`, `..._seed44` | baseline, colour + depth + thermal |
| `L0_vis_rgbd_700demos_imagenet_seed43`, `..._seed44` | baseline, colour + depth |
| `L0_vis_rgbt_700demos_imagenet_seed43`, `..._seed44` | baseline, colour + thermal |
| `L0_vis_rgb_700demos_imagenet_seed43`, `..._seed44` | baseline, colour |
| `L0_vis_rgbdt_700demos_hcg_full_seed43`, `..._seed44` | hindsight cue gating, arm `full` |

They are started through the queue of the fusion-vis worktree, ahead of the trainings that have not started. A run
that fails is reported and not started again.

**Read-out, fixed now** (`python policy/readout_contacts_seeds.py > readout_seeds.txt`; the script was tested on
the six evaluations of the first round only).

1. Every run on `L0_vis_ell` against the same checkpoint on `L0_vis`, same episodes (sign test, p < 0.05); the macro
   average of each baseline and of the method over the three seeds on both levels.
2. The table of the baselines on `L0_vis_ell`, built as the frozen table of Stage 6
   ([`../grid_bench/README.md`](../grid_bench/README.md)): per cell the success of each baseline over its three
   seeds, the highest of them as the best baseline of the cell (ties go to the baseline with more sensors), and the
   macro averages. The cells whose best baseline is another one than on `L0_vis` are named.
3. The method (`hcg_full`, three seeds) under rules 4 and 5 of Stage 6 with the table of item 2, runs paired by seed
   on the same episodes and pooled: "better in nearly every case" **holds on `L0_vis_ell`** if no cell is below its
   best baseline with p < 0.05 and the macro average is above every baseline, and **does not hold** otherwise.
4. The method against the rgbdt baseline, and the baselines against each other in the order of Stage 6 (rgbdt, rgbd,
   rgbt, rgb), over all episodes on both levels.

**Not in this round.** The references of Stage 6 (single-sensor policies) are not evaluated on `L0_vis_ell`. The
policies stay the ones trained on demos collected with the default contacts. The runs of round 4 of the method and
of [`../lit_base/`](../lit_base/README.md) are evaluated on `L0_vis` as listed; their evaluation on `L0_vis_ell` is
not listed here.

### Results of round 2 (11/25 12:44 KST; seeds 42, 43 and 44, 1320 episodes per run, 120 per cell)

All ten evaluations ran (11:06 to 12:42 KST); none failed. The read-out is [`readout_seeds.txt`](readout_seeds.txt)
(`PYTHONPATH=$PWD/policy python policy/readout_contacts_seeds.py`), in the order fixed above.

**1. The two levels.** Each of the 15 checkpoints succeeds more often on `L0_vis_ell` than on `L0_vis` on the same
episodes (p from 5e-12 to 1e-46). Macro average over the three seeds, mean (standard deviation):

| run | `L0_vis` | `L0_vis_ell` | difference |
|---|---|---|---|
| rgbdt | 0.428 (0.013) | 0.587 (0.025) | +0.159 |
| rgbd | 0.397 (0.009) | 0.523 (0.010) | +0.126 |
| rgbt | 0.194 (0.004) | 0.304 (0.008) | +0.110 |
| rgb | 0.153 (0.011) | 0.237 (0.020) | +0.084 |
| `hcg_full` | 0.792 (0.004) | 0.904 (0.007) | +0.111 |

The numbers on `L0_vis` are the frozen ones of Stage 6.

**2. The table of the baselines on `L0_vis_ell`** (three seeds pooled, 360 episodes per cell and baseline):

| cell | sensors that show the glass | rgbdt | rgbd | rgbt | rgb | best baseline on `L0_vis_ell` | best baseline on `L0_vis` (frozen) |
|---|---|---|---|---|---|---|---|
| 0 | rgb+depth+thermal | 0.911 | 0.847 | 0.561 | 0.378 | rgbdt 0.911 | rgbdt 0.781 |
| 1 | depth+thermal | 0.908 | 0.858 | 0.275 | 0.081 | rgbdt 0.908 | rgbdt 0.758 |
| 2 | rgb+thermal | 0.433 | 0.289 | 0.542 | 0.442 | rgbt 0.542 | rgbt 0.397 |
| 3 | rgb+depth | 0.883 | 0.836 | 0.361 | 0.414 | rgbdt 0.883 | rgbd 0.700 |
| 4 | thermal | 0.231 | 0.061 | 0.236 | 0.056 | rgbt 0.236 | rgbt 0.189 |
| 5 | depth | 0.903 | 0.886 | 0.086 | 0.092 | rgbdt 0.903 | rgbd 0.697 |
| 6 | rgb | 0.256 | 0.203 | 0.364 | 0.414 | rgb 0.414 | rgb 0.256 |
| 7 | rgb (faint) | 0.192 | 0.186 | 0.294 | 0.314 | rgb 0.314 | rgb 0.181 |
| 8 | depth (faint) | 0.783 | 0.742 | 0.100 | 0.078 | rgbdt 0.783 | rgbdt 0.486 |
| 9 | thermal (faint) | 0.161 | 0.106 | 0.172 | 0.072 | rgbt 0.172 | rgbt 0.094 |
| 10 | all three (faint) | 0.794 | 0.742 | 0.356 | 0.267 | rgbdt 0.794 | rgbdt 0.503 |
| macro average | | 0.587 | 0.523 | 0.304 | 0.237 | 0.624 | 0.458 |

The best baseline is another one in two cells (3 and 5: rgbdt instead of rgbd). On `L0_vis_ell` rgbdt is the best
baseline of every cell in which depth shows the glass, rgbt of the three in which thermal does and depth does not,
and rgb of the two in which only the colour camera does. The baselines where depth shows the glass are at 0.78 to
0.91; where it does not they stay at 0.17 to 0.54.

**3. The method (`hcg_full`, three seeds) under rules 4 and 5 of Stage 6, with the table of item 2.**

| cell | sensors that show the glass | seed 42 | seed 43 | seed 44 | pooled | best baseline of the cell | only the method succeeds / only the baseline | p | |
|---|---|---|---|---|---|---|---|---|---|
| 0 | rgb+depth+thermal | 0.983 | 0.983 | 0.983 | 0.983 | 0.911 (rgbdt) | 29 / 3 | 2.6e-06 | above |
| 1 | depth+thermal | 0.942 | 0.958 | 0.958 | 0.953 | 0.908 (rgbdt) | 29 / 13 | 0.02 | above |
| 2 | rgb+thermal | 0.950 | 0.958 | 0.983 | 0.964 | 0.542 (rgbt) | 159 / 7 | 1.4e-38 | above |
| 3 | rgb+depth | 0.958 | 0.983 | 0.967 | 0.969 | 0.883 (rgbdt) | 41 / 10 | 1.5e-05 | above |
| 4 | thermal | 0.833 | 0.825 | 0.858 | 0.839 | 0.236 (rgbt) | 222 / 5 | 4.6e-59 | above |
| 5 | depth | 0.875 | 0.875 | 0.917 | 0.889 | 0.903 (rgbdt) | 31 / 36 | 0.63 | |
| 6 | rgb | 0.958 | 0.950 | 0.967 | 0.958 | 0.414 (rgb) | 197 / 1 | 9.9e-58 | above |
| 7 | rgb (faint) | 0.950 | 0.958 | 0.967 | 0.958 | 0.314 (rgb) | 234 / 2 | 5.1e-67 | above |
| 8 | depth (faint) | 0.817 | 0.808 | 0.800 | 0.808 | 0.783 (rgbdt) | 63 / 54 | 0.46 | |
| 9 | thermal (faint) | 0.675 | 0.683 | 0.708 | 0.689 | 0.172 (rgbt) | 197 / 11 | 3.1e-45 | above |
| 10 | all three (faint) | 0.933 | 0.942 | 0.908 | 0.928 | 0.794 (rgbdt) | 57 / 9 | 1.2e-09 | above |
| macro average | | 0.898 | 0.902 | 0.911 | 0.904 | 0.624 | | | |

No cell is below its best baseline with p < 0.05; nine are above and two are not separated (5 and 8, the cells in
which depth alone shows the glass). The macro average is 0.904 (lowest seed 0.898, highest 0.911) against 0.587 for
the highest baseline, and above each baseline on the same episodes (p < 1e-250). **"Better in nearly every case"
holds on `L0_vis_ell`**, as it does on `L0_vis`.

**4. The differences on both levels** (runs paired by seed, pooled):

| pair | `L0_vis` | `L0_vis_ell` |
|---|---|---|
| `hcg_full` against rgbdt | 0.792 / 0.428, +0.364 (p = 5.9e-276) | 0.904 / 0.587, +0.317 (p = 5.6e-251) |
| rgbdt against rgbd | 0.428 / 0.397, +0.031 (p = 7.4e-05) | 0.587 / 0.523, +0.064 (p = 8.2e-19) |
| rgbd against rgbt | 0.397 / 0.194, +0.203 (p = 1.2e-93) | 0.523 / 0.304, +0.219 (p = 1.4e-92) |
| rgbt against rgb | 0.194 / 0.153, +0.042 (p = 1.6e-08) | 0.304 / 0.237, +0.067 (p = 4.8e-16) |

**What this says.**

- The pick table of the paper on `L0_vis_ell`: baselines 0.587 (rgbdt), 0.523 (rgbd), 0.304 (rgbt), 0.237 (rgb), the
  best baseline of every cell 0.624 on average, the method 0.904 with a standard deviation of 0.007 over the seeds.
- The verdict of Stage 6 is the same on both levels, and so is the order of the baselines. The method's margin over
  the rgbdt baseline is 0.32 instead of 0.36: the baseline gains more from the contact model (+0.16) than the method
  (+0.11), which is at 0.95 or above in six cells.
- The cells in which the method is not ahead are the same on both levels and on the place level: those in which
  depth alone shows the glass. The cells in which it stays lowest are 9 (thermal, faint: 0.689), 8 (0.808) and 4
  (0.839).

**What it does not say.** The policies are those trained on demos collected with the default contacts; the references
of Stage 6 were not evaluated on `L0_vis_ell`; the controls of the method and the baselines from the literature
follow in round 3.

## Round 3: the controls of the method, the seeds of round 4 and the baselines from the literature on `L0_vis_ell` (decision of the user 11/25; listed 11/25 12:02 KST, before any of these evaluations starts)

**Decision of the user (11/25).** The table of the paper is on `L0_vis_ell` (round 2). The runs that stand in that
table or carry its claims are therefore evaluated on `L0_vis_ell` too: the baselines from the literature
([`../lit_base/README.md`](../lit_base/README.md)), the seeds of `cue` and of the one-network control (round 4 of
[`../cue_gate/README.md`](../cue_gate/README.md)), and the seed-42 controls of rounds 1 and 2 of the method, on which
the statement "what the method is made of" rests. Evaluations only (`bash policy/eval_contacts.sh <run>`:
`j_cells11` and `k_cells11_heldout`, 660 episodes each, last checkpoint, about 35 minutes per run). Nothing is
retrained; every training and every evaluation on `L0_vis` stays as listed.

**Evaluations (18).**

| runs | what they are | when |
|---|---|---|
| `L0_vis_rgbdt_700demos_hcg_cue_early`, `..._hcg_cue_nogate`, `..._hcg_cue_mapgate`, `..._hcg_blind`, `L0_vis_rgbdt_700demos_dp_aux` | the seed-42 controls of rounds 1 and 2 of the method (one network, no gate, gate on map numbers, no images, the label as an auxiliary loss) | existing checkpoints: after the ten evaluations of round 2, before the trainings that have not started |
| `L0_vis_rgbdt_400demos_nosingle_hcg_cue`, `..._nosingle_hcg_cue_early` | seed 42 of the two runs trained without cells 4 to 9 (round 3b of the method) | the same |
| `L0_vis_rgbdt_700demos_mulsa` | self-attention fusion, seed 42 (finished 11/25 11:30 KST) | the same |
| `L0_vis_rgbdt_700demos_msbot`, `L0_vis_rgbdt_700demos_dropout_imagenet` | the two other baselines from the literature, seed 42 | not trained yet: in the queue line of the run, after its training and its evaluations on `L0_vis` (`... && bash policy/eval_contacts.sh <run>`) |
| `L0_vis_rgbdt_700demos_hcg_cue_seed43`, `..._seed44`, `L0_vis_rgbdt_700demos_hcg_cue_early_seed43`, `..._seed44`, `L0_vis_rgbdt_400demos_nosingle_hcg_cue_seed43`, `..._seed44`, `L0_vis_rgbdt_400demos_nosingle_hcg_cue_early_seed43`, `..._seed44` | the eight runs of round 4 of the method | the same |

`msbot` is trained in the fusion-vis worktree (its code is there), so its evaluation runs there and its results are in
that worktree's `results/hard_scenario/contact_check/`. The trainings and their evaluations on `L0_vis` are not
changed by the added step; a training that fails is not evaluated. The three kinds of checkpoint that
`eval_contacts.sh` had not loaded before (`mulsa`, `dp_aux`, `hcg_blind`) were tried with two episodes each on
`L0_vis_ell` before the listing (11/25, results discarded): all three load and run.

**Known when this is listed.** On `L0_vis_ell`: the seed-42 evaluations of the four plain baselines, of `hcg_cue`
(0.914) and of `hcg_full` (0.898) from the first round, and three of the ten evaluations of round 2. From the test of
the script below: `full` against `cue` on `L0_vis_ell`, seed 42, is 0.898 against 0.914 (30 / 52, p = 0.02). None of
the 18 evaluations above has started.

**Read-out, fixed now.** All on the same episodes, sign test, p < 0.05; the read-outs are those fixed for `L0_vis`,
applied to the evaluations on `L0_vis_ell`.

1. *What the method is made of* (`PYTHONPATH=$PWD/policy python policy/readout_contacts_round3.py >
   readout_round3.txt`, part A; seed 42). The pairs of rounds 1 and 2 of the method on `L0_vis_ell`, next to the same
   pair on `L0_vis`: `full` against `cue` (gating the image features), `cue` against `blind` (the images), and `cue`
   against each control (one network, gate on map numbers, no gate, the label as an auxiliary loss). A part of the
   method **matters on `L0_vis_ell`** when `cue` is above the control without it with p < 0.05, as in round 2. Next
   to it: the share of the gain of `cue` over the plain rgbdt baseline that each run keeps, and the cells in which it
   is below the best plain baseline of the cell (seed-42 runs).
2. *The seeds* (`python policy/readout_round4.py --root results/hard_scenario/contact_check >
   readout_round4_ell.txt`): settings 1 (training cells) and 3 (trained without cells 4 to 9) of round 4 with its
   rule, unchanged: the difference between `cue` and the one network **holds over the seeds** when the pooled test is
   significant and `cue` is above in each of the three seeds, **does not hold** when the pooled test is not
   significant, and is **mixed** otherwise. Setting 2 (the 15 new combinations) is not evaluated on `L0_vis_ell`.
3. *The baselines from the literature* (part B of `readout_contacts_round3.py`; seed 42): items 1, 2 and 4 of the
   read-out of `lit_base` on `L0_vis_ell`: against the plain rgbdt baseline over all episodes and per cell, per cell
   against the cell's best plain baseline (seed-42 runs on `L0_vis_ell`), against `hcg_full` and `hcg_cue`, and where
   the gripper first closes. Run where the results of all three are present.

The scripts were tested on the run folders of `L0_vis` standing in for the evaluations that do not exist yet.

**Not in this round.** The 15 new combinations on `L0_vis_ell`; the runs of [`../fusion_vis/`](../fusion_vis/README.md)
on `L0_vis_ell`; the references of Stage 6; seeds 43 and 44 of the baselines from the literature (they follow after
the first reading, as listed there, and are then evaluated on both levels).

### Results of round 3, first part: the seed-42 controls and self-attention fusion (11/25 13:58 KST; 1320 episodes per run, 120 per cell)

The eight evaluations of existing checkpoints ran (12:15 to 13:48 KST); none failed. The read-out is
[`readout_round3.txt`](readout_round3.txt) (`PYTHONPATH=$PWD/policy python policy/readout_contacts_round3.py`). Item 1
is complete. Item 2 needs seeds 43 and 44 (in training); what seed 42 shows is given without a verdict. Item 3 has
self-attention fusion only; the two other baselines from the literature are in training.

**1. What the method is made of** (seed 42, same episodes on each level).

| what differs | first | second | `L0_vis`: first / second | only the first succeeds / only the second | p | `L0_vis_ell`: first / second | only the first / only the second | p | on `L0_vis_ell` the first is |
|---|---|---|---|---|---|---|---|---|---|
| gating the image features | `full` | `cue` | 0.789 / 0.795 | 98 / 107 | 0.58 | 0.898 / 0.914 | 30 / 52 | 0.02 | below |
| the images | `cue` | `blind` | 0.795 / 0.783 | 112 / 95 | 0.27 | 0.914 / 0.888 | 64 / 29 | 0.00037 | above |
| maps per sensor with the gate, against one network | `cue` | one network | 0.795 / 0.730 | 215 / 128 | 3.1e-06 | 0.914 / 0.911 | 75 / 70 | 0.74 | not separated |
| a gate that looks at the image, against a gate on map numbers | `cue` | gate on map numbers | 0.795 / 0.695 | 230 / 97 | 1.3e-13 | 0.914 / 0.825 | 163 / 45 | 6.7e-17 | above |
| a gate among the sensors, against none | `cue` | no gate | 0.795 / 0.502 | 446 / 58 | 3.4e-75 | 0.914 / 0.655 | 379 / 36 | 2.6e-73 | above |
| the cue in the conditioning, against the label as an auxiliary loss | `cue` | auxiliary loss | 0.795 / 0.226 | 776 / 24 | 1.7e-195 | 0.914 / 0.439 | 655 / 27 | 1.8e-157 | above |

By the rule fixed above, on `L0_vis_ell` these parts **matter**: a gate among the sensors against none, a gate that
looks at the image against one on map numbers, the cue in the conditioning against an auxiliary loss, and the images
(`cue` above `blind`; on `L0_vis` the two were not separated). These do **not**: the maps per sensor with the gate
against one network over all sensors (0.914 against 0.911; on `L0_vis` `cue` was 6.5 points above), and gating the
image features (`full` is below `cue` on `L0_vis_ell`).

The share of the gain of `cue` over the plain rgbdt baseline that a run keeps, and the cells (p < 0.05):

| run | `L0_vis` | kept | `L0_vis_ell` | kept | cells below their best plain baseline on `L0_vis_ell` | cells in which it differs from `cue` on `L0_vis_ell`: above / below |
|---|---|---|---|---|---|---|
| `full` | 0.789 | 0.98 | 0.898 | 0.95 | none | none / 8 |
| `cue` | 0.795 | 1.00 | 0.914 | 1.00 | none | |
| `blind` | 0.783 | 0.97 | 0.888 | 0.93 | none | none / 5, 8 |
| one network | 0.730 | 0.83 | 0.911 | 0.99 | none | 4 / 7 |
| gate on map numbers | 0.695 | 0.74 | 0.825 | 0.75 | none | none / 4, 6, 7, 8, 9 |
| no gate | 0.502 | 0.23 | 0.655 | 0.27 | 5, 8 | none / 0, 2, 4, 5, 6, 7, 8, 9, 10 |
| auxiliary loss | 0.226 | -0.49 | 0.439 | -0.34 | 0, 1, 3, 5, 8, 10 | none / all eleven |
| plain rgbdt | 0.414 | 0 | 0.560 | 0 | | |

Success per cell on `L0_vis_ell` (seed 42; `cells.json` of every run):

| cell | sensors that show the glass | `full` | `cue` | `blind` | one network | gate on map numbers | no gate | auxiliary loss | plain rgbdt |
|---|---|---|---|---|---|---|---|---|---|
| 0 | rgb+depth+thermal | 0.983 | 1.000 | 0.992 | 0.967 | 0.975 | 0.950 | 0.625 | 0.900 |
| 1 | depth+thermal | 0.942 | 0.933 | 0.958 | 0.967 | 0.933 | 0.917 | 0.608 | 0.892 |
| 2 | rgb+thermal | 0.950 | 0.983 | 0.967 | 0.950 | 0.975 | 0.817 | 0.650 | 0.350 |
| 3 | rgb+depth | 0.958 | 0.958 | 0.992 | 0.933 | 0.933 | 0.917 | 0.600 | 0.858 |
| 4 | thermal | 0.833 | 0.833 | 0.833 | 0.950 | 0.658 | 0.483 | 0.192 | 0.250 |
| 5 | depth | 0.875 | 0.883 | 0.775 | 0.950 | 0.800 | 0.642 | 0.475 | 0.875 |
| 6 | rgb | 0.958 | 0.967 | 0.942 | 0.942 | 0.825 | 0.408 | 0.367 | 0.242 |
| 7 | rgb (faint) | 0.950 | 0.958 | 0.967 | 0.892 | 0.792 | 0.383 | 0.267 | 0.158 |
| 8 | depth (faint) | 0.817 | 0.875 | 0.758 | 0.817 | 0.717 | 0.508 | 0.308 | 0.767 |
| 9 | thermal (faint) | 0.675 | 0.717 | 0.658 | 0.725 | 0.508 | 0.342 | 0.233 | 0.117 |
| 10 | all three (faint) | 0.933 | 0.950 | 0.925 | 0.925 | 0.958 | 0.833 | 0.500 | 0.750 |
| macro average | | 0.898 | 0.914 | 0.888 | 0.911 | 0.825 | 0.655 | 0.439 | 0.560 |

**2. The seeds: what seed 42 shows, no verdict yet** (`python policy/readout_round4.py --root
results/hard_scenario/contact_check`, run with the one seed present; `readout_round4_ell.txt` is written when the three
seeds are).

| setting | cells | level | `cue` | one network | only `cue` / only the one network | p |
|---|---|---|---|---|---|---|
| 1. training cells | all eleven | `L0_vis` | 0.795 | 0.730 | 215 / 128 | 3.1e-06 |
| | | `L0_vis_ell` | 0.914 | 0.911 | 75 / 70 | 0.74 |
| 3. trained without cells 4 to 9 | the cells left out (4 to 9) | `L0_vis` | 0.678 | 0.550 | 179 / 87 | 1.8e-08 |
| | | `L0_vis_ell` | 0.831 | 0.749 | 126 / 67 | 2.6e-05 |
| | the cells of the training set (0, 1, 2, 3, 10) | `L0_vis` | 0.883 | 0.875 | 47 / 42 | 0.67 |
| | | `L0_vis_ell` | 0.958 | 0.955 | 16 / 14 | 0.86 |

In the cells the two runs were not trained on, `cue` stays above the one network on `L0_vis_ell` (8 points); in the
cells they were trained on, in either setting, the two are level there.

**3. Self-attention fusion (MulSA), a first reading** (part B; the item is read again when the three baselines from
the literature are evaluated).

- Against the plain rgbdt baseline: 0.564 against 0.560 (120 / 114, p = 0.74); on `L0_vis` 0.412 against 0.414. No
  cell differs from plain rgbdt. Against the best plain baseline of the cell it is below in cell 6 (0.242 against
  rgb 0.450) and above in none.
- Against the method: 0.564 against 0.898 (`full`; 46 / 486) and 0.914 (`cue`; 36 / 498); below `full` in eight
  cells and `cue` in nine, above in none.
- The first close of the gripper is within 1.5 cm of the glass in 0.471 of the episodes (plain rgbdt 0.461, `full`
  and `cue` 0.870); the success after such a close is 0.923 (plain 0.929, `full` 0.979, `cue` 0.996).

**Next to the read-out, not a test: the cue's error on both levels**
([`cue_error_bins_ell.txt`](cue_error_bins_ell.txt): the first table of `python policy/cue_error_bins.py --root
results/hard_scenario/contact_check`; the script then stops at the 15 new combinations, which are not evaluated on
`L0_vis_ell`; the table of `L0_vis` is [`../cue_gate/cue_error_bins.txt`](../cue_gate/cue_error_bins.txt)). Share of
the episodes / success, by the distance in the table plane between the cue's grasp point and the glass:

| run | level | 0 to 0.5 cm | 0.5 to 1.0 cm | 1.0 to 1.5 cm | over 1.5 cm |
|---|---|---|---|---|---|
| `cue` | `L0_vis` | 0.53 / 0.93 | 0.28 / 0.80 | 0.09 / 0.57 | 0.09 / 0.22 |
| | `L0_vis_ell` | 0.53 / 0.99 | 0.28 / 0.96 | 0.09 / 0.88 | 0.09 / 0.41 |
| one network | `L0_vis` | 0.39 / 0.89 | 0.38 / 0.77 | 0.12 / 0.60 | 0.12 / 0.19 |
| | `L0_vis_ell` | 0.39 / 0.98 | 0.37 / 0.97 | 0.12 / 0.92 | 0.12 / 0.49 |
| gate on map numbers | `L0_vis` | 0.46 / 0.89 | 0.23 / 0.80 | 0.09 / 0.69 | 0.22 / 0.19 |
| | `L0_vis_ell` | 0.46 / 0.99 | 0.23 / 0.95 | 0.09 / 0.92 | 0.22 / 0.32 |
| no gate | `L0_vis` | 0.25 / 0.78 | 0.23 / 0.63 | 0.13 / 0.50 | 0.39 / 0.24 |
| | `L0_vis_ell` | 0.26 / 0.91 | 0.23 / 0.81 | 0.13 / 0.71 | 0.39 / 0.38 |

The cues are those of the same checkpoints on the same scenes, so the shares are the same on both levels. The one
network puts nearly as many cues within 1.5 cm of the glass as `cue` (0.88 against 0.91) but fewer within 0.5 cm
(0.39 against 0.53). For these two runs a cue 0.5 to 1.5 cm off fails in about a quarter of the episodes with the
default contacts (0.25 and 0.27) and in about one in twenty with the contact model of the place level (0.06 and
0.04). The precision in which the maps per sensor were ahead is what this contact model no longer asks for. The gate
on map numbers and the run without a gate put more cues beyond 1.5 cm (0.22 and 0.39 against 0.09), and most of
those fail on both levels.

**What this says.**

- *A choice among the sensors is needed, on both levels.* Without it (the maps averaged) the run keeps a quarter of
  the gain and is below the plain baselines in the two cells in which depth alone shows the glass; a gate that sees
  only the numbers of the maps keeps three quarters.
- *How the choice is made is not settled on the level of the paper's table.* One network over all three sensors,
  with no gate to read, forms the cue as well as the maps per sensor with the gate in the cells both were trained
  on (0.911 against 0.914, seed 42). What `cue` keeps over it at seed 42 is in the combinations of sensors not in
  the training set (0.831 against 0.749 in the cells left out). The verdicts of item 2 need the three seeds.
- *Gating the image features adds nothing*: not separated on `L0_vis` and on the place level, below on `L0_vis_ell`.
- *The images matter on `L0_vis_ell`* (2.6 points, cells 5 and 8), which `L0_vis` did not show.
- *Self-attention fusion is a plain rgbdt policy by its results* on both levels.

**What it does not say.** One seed each. The new combinations (setting 2) are not evaluated on `L0_vis_ell`. That the
one network catches up is read from the cue's error after the fact; no run was made to test it.

### Results of round 3, items 2 and 3 (11/26)

**Item 2, the seeds of `cue` and of the one network on `L0_vis_ell`** ([`readout_round4_ell.txt`](readout_round4_ell.txt);
reported with round 4 in [`../cue_gate/README.md`](../cue_gate/README.md), "Results of round 4"). Training cells: 0.915
against 0.909 over the three seeds (239 / 213, p = 0.24): the difference **does not hold**. Trained without cells 4 to
9, the cells left out: 0.829 against 0.740 (384 / 192, p = 9.5e-16), `cue` above in each seed: it **holds over the
seeds**; in the cells of that training set 0.956 against 0.954, it does not hold. The new combinations (setting 2) are
evaluated on `L0_vis_ell` in round 5 of that page.

**Item 3, the baselines from the literature on `L0_vis_ell`** is reported in
[`../lit_base/README.md`](../lit_base/README.md), "Results of the first reading" (seed 42: self-attention fusion 0.564
and sensor dropout 0.556 against plain rgbdt 0.560, MS-Bot 0.273; `readout_ell.txt` there).

