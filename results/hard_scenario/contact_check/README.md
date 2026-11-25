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
