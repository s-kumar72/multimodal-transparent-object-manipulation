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
- **The teacher on `L0_vis_ell`:** see "Teacher" below (filled in when the check has run, before the evaluations are
  read).

## Runs (listed 11/25 01:35 KST, before any of them starts)

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
