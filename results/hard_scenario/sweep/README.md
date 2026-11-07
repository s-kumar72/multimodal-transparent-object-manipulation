# Hard-scenario sweep: L4b results

L4b × `rgb` / `rgbd` / `rgbdt`, with the settings from the plan in the top README: 200 demos, 101 epochs, rollouts
on test set (a) every 25 epochs, then the last checkpoint on test sets (a) and (g), 50 episodes each. One seed per
run. The runs used the same hydra overrides and eval command as `scripts/aws/sweep.sh`, on one RTX A6000 (40-60 min per
run including evaluation, 2026-11-07). The teacher succeeded in all 200 demos.

**The L0 runs have not been done.** The plan was to run them only if L4b × `rgbdt` reached 0.80 on test set (a).
It reached 0.14, so they are on hold until we decide on the level design.

## Success rate

| run | (a) in-distribution | (g) held-out | rollouts on (a) during training, epoch 0 / 25 / 50 / 75 / 100 |
|---|---|---|---|
| rgb | 0.00 (0/50) | 0.00 (0/50) | 0.00 / 0.02 / 0.04 / 0.04 / 0.04 |
| rgbd | 0.08 (4/50) | 0.10 (5/50) | 0.00 / 0.02 / 0.00 / 0.06 / 0.12 |
| rgbdt | 0.14 (7/50) | 0.10 (5/50) | 0.02 / 0.02 / 0.12 / 0.18 / 0.16 |

The targets were: rgb ≤ 0.80 (met), rgb < rgbd < 0.90 (met), rgbdt ≥ 0.90 (not met).

On (a), rgb vs rgbdt is a real difference (Fisher exact test, two-sided p = 0.013). rgbd vs rgbdt (p = 0.53) and
rgb vs rgbd (p = 0.12) are not. The epoch-100 rollout and the final (a) evaluation use the same checkpoint and test
set but differ by one or two episodes, which shows how noisy these numbers are.

Per object on (a), 10 episodes each:

| run | bottle | flask | jar | tumbler | wineglass |
|---|---|---|---|---|---|
| rgb | 0 | 0 | 0 | 0 | 0 |
| rgbd | 1 | 0 | 1 | 0 | 2 |
| rgbdt | 2 | 2 | 2 | 0 | 1 |

## How the episodes fail (test set (a))

The outcome definitions are in `policy/glass_gym_env.py` (`outcome`).

| outcome | rgb | rgbd | rgbdt |
|---|---|---|---|
| success | 0.00 | 0.08 | 0.14 |
| missed_grasp: closed within 8 cm of the target, no lift | 0.32 | 0.58 | 0.52 |
| other: closed away from every object | 0.54 | 0.18 | 0.16 |
| dropped | 0.08 | 0.06 | 0.12 |
| wrong_height | 0.02 | 0.06 | 0.06 |
| wrong_object | 0.04 | 0.02 | 0.00 |
| timeout | 0.00 | 0.02 | 0.00 |

## What this suggests

- **Depth changes how the policy fails.** With rgb alone, more than half of the episodes close away from every object,
  so the policy does not find the glass. With depth, most episodes close within 8 cm of the target and still don't
  lift it. The remaining problem is grasp precision rather than finding the target.
- **The input carries the information, but the policy doesn't use it.** On L4b, the probe (`../probe.md`) places the
  target within 3 px in 0.94 of the frames with rgbdt (0.73 with rgb). With 200 demos, the policy does not turn this
  into precise grasps.
- **rgbd and rgbdt overfit.** Validation loss was lowest at epoch 45 for rgbdt (0.0123) and epoch 67 for rgbd
  (0.0096), and ended at 0.0239 and 0.0157. Training loss kept falling, to 0.0027 and 0.0034. rgb did not overfit
  (validation 0.0047, training 0.0044 at epoch 100). Validation loss does not predict success well here: rgbdt's best
  rollout came at epoch 75, after its validation minimum.
- **The target is small.** With the default camera, the target covers about 0.26% of the 96×96 image, roughly 24 px.
  `L4b_narrow` (about 2.3%) would make it larger, but it also removes most of the modality gap in the probe. There,
  rgb already places the target within 3 px in 0.94 of the frames and gets 0.90 object-type accuracy, against 0.98
  and 0.98 for rgbdt.

## Open questions

- **Is L4b too hard, or is 200 demos / 101 epochs too little?** The earlier main-scene rgb result (0.96) used 500
  demos, so it does not answer this. L0 × rgbdt with the same budget (about 1 hour on an A6000) would separate the two.
- **Test set (f), the thermal crossover, was not evaluated** because it was not in the plan.
- **Each run's `latest.ckpt` is kept outside git.** It is 4.6 GB, so more test sets or episodes can be run without
  retraining.

## Files

The layout is the one `scripts/aws/sweep.sh` writes. Each `L4b_<variant>/` has `train_logs.json.txt` and `curve.json`,
and `eval_log.json` and `episodes.json` for each test set.
