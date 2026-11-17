# Round 2 summary: `python policy/summarize_runs.py table` on the ten Round 2 run folders (2026-11-14)


## L0_distract_rgb_500demos

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.20 (10/50) | 0.11-0.33 | 0.56 | 0.02 | 0.02 | 0.20 | 0.00 | 0.00 |
| g_heldout_all | 0.24 (12/50) | 0.14-0.37 | 0.46 | 0.06 | 0.06 | 0.16 | 0.02 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.04, ep50 0.12, ep75 0.24, ep100 0.30
val loss: lowest 0.0026 at epoch 81, last 0.0035 at epoch 100; train loss last 0.0017

success by condition on a_in_distribution (successes/episodes):
- object: bottle 1/10, flask 1/10, jar 4/10, tumbler 4/10, wineglass 0/10
- background: 0 10/50
- lighting: 0 10/50
- distractors: 1 4/16, 2 3/19, 3 3/15
- look-alike: 0 10/50
- thermal crossover: 0 9/44, 1 1/6
- liquid: empty 10/50
- depth see-through: 0 10/50

## L0_table_rgb_500demos

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.00 (0/50) | 0.00-0.07 | 0.36 | 0.52 | 0.04 | 0.08 | 0.00 | 0.00 |
| g_heldout_all | 0.04 (2/50) | 0.01-0.13 | 0.20 | 0.70 | 0.06 | 0.00 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.02, ep50 0.02, ep75 0.02, ep100 0.06
val loss: lowest 0.0024 at epoch 96, last 0.0024 at epoch 100; train loss last 0.0025

success by condition on a_in_distribution (successes/episodes):
- object: bottle 0/10, flask 0/10, jar 0/10, tumbler 0/10, wineglass 0/10
- background: 1 0/14, 2 0/7, 4 0/13, 5 0/16
- lighting: 0 0/50
- distractors: 0 0/50
- look-alike: 0 0/50
- thermal crossover: 0 0/44, 1 0/6
- liquid: empty 0/50
- depth see-through: 0 0/50

## L0_light_rgb_500demos

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.04 (2/50) | 0.01-0.13 | 0.18 | 0.62 | 0.14 | 0.02 | 0.00 | 0.00 |
| g_heldout_all | 0.04 (2/50) | 0.01-0.13 | 0.26 | 0.66 | 0.02 | 0.02 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.00, ep50 0.00, ep75 0.04, ep100 0.06
val loss: lowest 0.0023 at epoch 96, last 0.0024 at epoch 100; train loss last 0.0025

success by condition on a_in_distribution (successes/episodes):
- object: bottle 1/10, flask 1/10, jar 0/10, tumbler 0/10, wineglass 0/10
- background: 0 2/50
- lighting: 0 0/14, 1 0/14, 2 0/11, 3 2/11
- distractors: 0 2/50
- look-alike: 0 2/50
- thermal crossover: 0 1/44, 1 1/6
- liquid: empty 2/50
- depth see-through: 0 2/50

## L0_scene_rgb_500demos

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.04 (2/50) | 0.01-0.13 | 0.22 | 0.68 | 0.04 | 0.02 | 0.00 | 0.00 |
| g_heldout_all | 0.04 (2/50) | 0.01-0.13 | 0.20 | 0.74 | 0.02 | 0.00 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.00, ep50 0.02, ep75 0.04, ep100 0.02
val loss: lowest 0.0023 at epoch 96, last 0.0025 at epoch 100; train loss last 0.0025

success by condition on a_in_distribution (successes/episodes):
- object: bottle 0/10, flask 1/10, jar 1/10, tumbler 0/10, wineglass 0/10
- background: 1 1/14, 2 0/7, 4 0/13, 5 1/16
- lighting: 0 2/50
- distractors: 1 1/16, 2 1/19, 3 0/15
- look-alike: 0 2/50
- thermal crossover: 0 2/44, 1 0/6
- liquid: empty 2/50
- depth see-through: 0 2/50

## L0_distract_g_rgb_500demos

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.18 (9/50) | 0.10-0.31 | 0.50 | 0.06 | 0.06 | 0.18 | 0.00 | 0.02 |
| g_heldout_all | 0.34 (17/50) | 0.22-0.48 | 0.44 | 0.02 | 0.06 | 0.12 | 0.02 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.04, ep50 0.06, ep75 0.24, ep100 0.32
val loss: lowest 0.0025 at epoch 81, last 0.0031 at epoch 100; train loss last 0.0017

success by condition on a_in_distribution (successes/episodes):
- object: bottle 3/10, flask 0/10, jar 2/10, tumbler 4/10, wineglass 0/10
- background: 0 9/50
- lighting: 0 9/50
- distractors: 1 4/16, 2 1/19, 3 4/15
- look-alike: 0 9/50
- thermal crossover: 0 7/44, 1 2/6
- liquid: empty 9/50
- depth see-through: 0 9/50

## L0_distract_g_rgbd_500demos

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.46 (23/50) | 0.33-0.60 | 0.40 | 0.00 | 0.00 | 0.14 | 0.00 | 0.00 |
| g_heldout_all | 0.44 (22/50) | 0.31-0.58 | 0.36 | 0.04 | 0.10 | 0.06 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.34, ep50 0.30, ep75 0.42, ep100 0.36
val loss: lowest 0.0025 at epoch 93, last 0.0026 at epoch 100; train loss last 0.0026

success by condition on a_in_distribution (successes/episodes):
- object: bottle 4/10, flask 4/10, jar 4/10, tumbler 5/10, wineglass 6/10
- background: 0 23/50
- lighting: 0 23/50
- distractors: 1 6/16, 2 5/19, 3 12/15
- look-alike: 0 23/50
- thermal crossover: 0 19/44, 1 4/6
- liquid: empty 23/50
- depth see-through: 0 23/50

## L0_distract_g_rgbdt_500demos

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.54 (27/50) | 0.40-0.67 | 0.24 | 0.00 | 0.08 | 0.14 | 0.00 | 0.00 |
| g_heldout_all | 0.38 (19/50) | 0.26-0.52 | 0.38 | 0.02 | 0.08 | 0.14 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.36, ep50 0.36, ep75 0.46, ep100 0.48
val loss: lowest 0.0028 at epoch 87, last 0.0029 at epoch 100; train loss last 0.0021

success by condition on a_in_distribution (successes/episodes):
- object: bottle 6/10, flask 3/10, jar 6/10, tumbler 5/10, wineglass 7/10
- background: 0 27/50
- lighting: 0 27/50
- distractors: 1 9/16, 2 7/19, 3 11/15
- look-alike: 0 27/50
- thermal crossover: 0 24/44, 1 3/6
- liquid: empty 27/50
- depth see-through: 0 27/50

## L0_distract_g_rgbt_500demos

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.40 (20/50) | 0.28-0.54 | 0.34 | 0.02 | 0.22 | 0.02 | 0.00 | 0.00 |
| g_heldout_all | 0.50 (25/50) | 0.37-0.63 | 0.26 | 0.08 | 0.14 | 0.00 | 0.02 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.16, ep50 0.42, ep75 0.34, ep100 0.46
val loss: lowest 0.0041 at epoch 66, last 0.0055 at epoch 100; train loss last 0.0014

success by condition on a_in_distribution (successes/episodes):
- object: bottle 4/10, flask 2/10, jar 7/10, tumbler 2/10, wineglass 5/10
- background: 0 20/50
- lighting: 0 20/50
- distractors: 1 10/16, 2 4/19, 3 6/15
- look-alike: 0 20/50
- thermal crossover: 0 20/44, 1 0/6
- liquid: empty 20/50
- depth see-through: 0 20/50

## L0_distract_rgbd_500demos

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.84 (42/50) | 0.71-0.92 | 0.10 | 0.00 | 0.06 | 0.00 | 0.00 | 0.00 |
| g_heldout_all | 0.80 (40/50) | 0.67-0.89 | 0.08 | 0.00 | 0.06 | 0.06 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.72, ep50 0.86, ep75 0.92, ep100 0.90
val loss: lowest 0.0017 at epoch 100, last 0.0017 at epoch 100; train loss last 0.0019

success by condition on a_in_distribution (successes/episodes):
- object: bottle 9/10, flask 7/10, jar 10/10, tumbler 9/10, wineglass 7/10
- background: 0 42/50
- lighting: 0 42/50
- distractors: 1 12/16, 2 17/19, 3 13/15
- look-alike: 0 42/50
- thermal crossover: 0 37/44, 1 5/6
- liquid: empty 42/50
- depth see-through: 0 42/50

## L0_distract_rgbdt_500demos

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.88 (44/50) | 0.76-0.94 | 0.04 | 0.00 | 0.02 | 0.06 | 0.00 | 0.00 |
| g_heldout_all | 0.88 (44/50) | 0.76-0.94 | 0.04 | 0.00 | 0.04 | 0.04 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.64, ep50 0.80, ep75 0.88, ep100 0.88
val loss: lowest 0.0024 at epoch 96, last 0.0027 at epoch 100; train loss last 0.0014

success by condition on a_in_distribution (successes/episodes):
- object: bottle 9/10, flask 9/10, jar 10/10, tumbler 8/10, wineglass 8/10
- background: 0 44/50
- lighting: 0 44/50
- distractors: 1 15/16, 2 17/19, 3 12/15
- look-alike: 0 44/50
- thermal crossover: 0 39/44, 1 5/6
- liquid: empty 44/50
- depth see-through: 0 44/50
