<!-- python policy/summarize_runs.py table results/hard_scenario/benchmark/*/ -->

## L0_bench_rgb_500demos

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.40 (20/50) | 0.28-0.54 | 0.50 | 0.00 | 0.04 | 0.06 | 0.00 | 0.00 |
| g_heldout_all | 0.34 (17/50) | 0.22-0.48 | 0.44 | 0.04 | 0.14 | 0.02 | 0.02 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.08, ep50 0.18, ep75 0.36, ep100 0.46
val loss: lowest 0.0023 at epoch 96, last 0.0029 at epoch 100; train loss last 0.0015

success by condition on a_in_distribution (successes/episodes):
- object: bottle 3/10, flask 1/10, jar 7/10, tumbler 8/10, wineglass 1/10
- background: 0 20/50
- lighting: 0 20/50
- distractors: 1 7/16, 2 5/19, 3 8/15
- look-alike: 0 20/50
- thermal crossover: 0 18/44, 1 2/6
- liquid: empty 20/50
- depth see-through: 0 15/39, 1 5/11

## L0_bench_rgb_500demos_seed43

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.44 (22/50) | 0.31-0.58 | 0.42 | 0.02 | 0.08 | 0.04 | 0.00 | 0.00 |
| g_heldout_all | 0.34 (17/50) | 0.22-0.48 | 0.48 | 0.06 | 0.10 | 0.02 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.10, ep50 0.26, ep75 0.46, ep100 0.34
val loss: lowest 0.0026 at epoch 82, last 0.0032 at epoch 100; train loss last 0.0015

success by condition on a_in_distribution (successes/episodes):
- object: bottle 3/10, flask 2/10, jar 9/10, tumbler 6/10, wineglass 2/10
- background: 0 22/50
- lighting: 0 22/50
- distractors: 1 5/16, 2 9/19, 3 8/15
- look-alike: 0 22/50
- thermal crossover: 0 19/44, 1 3/6
- liquid: empty 22/50
- depth see-through: 0 17/39, 1 5/11

## L0_bench_rgbd_500demos

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.60 (30/50) | 0.46-0.72 | 0.26 | 0.08 | 0.02 | 0.04 | 0.00 | 0.00 |
| g_heldout_all | 0.64 (32/50) | 0.50-0.76 | 0.20 | 0.10 | 0.02 | 0.02 | 0.02 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.34, ep50 0.48, ep75 0.62, ep100 0.66
val loss: lowest 0.0023 at epoch 90, last 0.0027 at epoch 100; train loss last 0.0020

success by condition on a_in_distribution (successes/episodes):
- object: bottle 6/10, flask 6/10, jar 5/10, tumbler 5/10, wineglass 8/10
- background: 0 30/50
- lighting: 0 30/50
- distractors: 1 10/16, 2 10/19, 3 10/15
- look-alike: 0 30/50
- thermal crossover: 0 25/44, 1 5/6
- liquid: empty 30/50
- depth see-through: 0 30/39, 1 0/11

## L0_bench_rgbd_500demos_seed43

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.66 (33/50) | 0.52-0.78 | 0.24 | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 |
| g_heldout_all | 0.60 (30/50) | 0.46-0.72 | 0.20 | 0.10 | 0.06 | 0.04 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.26, ep50 0.26, ep75 0.62, ep100 0.60
val loss: lowest 0.0024 at epoch 88, last 0.0029 at epoch 100; train loss last 0.0020

success by condition on a_in_distribution (successes/episodes):
- object: bottle 5/10, flask 8/10, jar 7/10, tumbler 8/10, wineglass 5/10
- background: 0 33/50
- lighting: 0 33/50
- distractors: 1 10/16, 2 12/19, 3 11/15
- look-alike: 0 33/50
- thermal crossover: 0 29/44, 1 4/6
- liquid: empty 33/50
- depth see-through: 0 32/39, 1 1/11

## L0_bench_rgbdt_500demos

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.58 (29/50) | 0.44-0.71 | 0.20 | 0.06 | 0.08 | 0.08 | 0.00 | 0.00 |
| g_heldout_all | 0.54 (27/50) | 0.40-0.67 | 0.26 | 0.08 | 0.06 | 0.06 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.36, ep50 0.44, ep75 0.60, ep100 0.62
val loss: lowest 0.0033 at epoch 84, last 0.0038 at epoch 100; train loss last 0.0013

success by condition on a_in_distribution (successes/episodes):
- object: bottle 3/10, flask 5/10, jar 9/10, tumbler 6/10, wineglass 6/10
- background: 0 29/50
- lighting: 0 29/50
- distractors: 1 7/16, 2 10/19, 3 12/15
- look-alike: 0 29/50
- thermal crossover: 0 26/44, 1 3/6
- liquid: empty 29/50
- depth see-through: 0 26/39, 1 3/11

## L0_bench_rgbdt_500demos_seed43

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.70 (35/50) | 0.56-0.81 | 0.24 | 0.04 | 0.02 | 0.00 | 0.00 | 0.00 |
| g_heldout_all | 0.60 (30/50) | 0.46-0.72 | 0.16 | 0.04 | 0.12 | 0.06 | 0.02 | 0.00 |

rollouts on (a) during training: ep0 0.02, ep25 0.26, ep50 0.42, ep75 0.66, ep100 0.52
val loss: lowest 0.0034 at epoch 81, last 0.0037 at epoch 100; train loss last 0.0013

success by condition on a_in_distribution (successes/episodes):
- object: bottle 5/10, flask 7/10, jar 8/10, tumbler 8/10, wineglass 7/10
- background: 0 35/50
- lighting: 0 35/50
- distractors: 1 12/16, 2 11/19, 3 12/15
- look-alike: 0 35/50
- thermal crossover: 0 32/44, 1 3/6
- liquid: empty 35/50
- depth see-through: 0 34/39, 1 1/11

## L0_bench_rgbt_500demos

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.56 (28/50) | 0.42-0.69 | 0.32 | 0.04 | 0.08 | 0.00 | 0.00 | 0.00 |
| g_heldout_all | 0.66 (33/50) | 0.52-0.78 | 0.20 | 0.04 | 0.06 | 0.04 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.20, ep50 0.40, ep75 0.52, ep100 0.56
val loss: lowest 0.0035 at epoch 66, last 0.0047 at epoch 100; train loss last 0.0011

success by condition on a_in_distribution (successes/episodes):
- object: bottle 5/10, flask 5/10, jar 8/10, tumbler 7/10, wineglass 3/10
- background: 0 28/50
- lighting: 0 28/50
- distractors: 1 11/16, 2 10/19, 3 7/15
- look-alike: 0 28/50
- thermal crossover: 0 24/44, 1 4/6
- liquid: empty 28/50
- depth see-through: 0 21/39, 1 7/11

## L0_bench_rgbt_500demos_seed43

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.42 (21/50) | 0.29-0.56 | 0.40 | 0.02 | 0.16 | 0.00 | 0.00 | 0.00 |
| g_heldout_all | 0.42 (21/50) | 0.29-0.56 | 0.42 | 0.04 | 0.08 | 0.04 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.04, ep25 0.12, ep50 0.24, ep75 0.40, ep100 0.38
val loss: lowest 0.0039 at epoch 81, last 0.0043 at epoch 100; train loss last 0.0012

success by condition on a_in_distribution (successes/episodes):
- object: bottle 3/10, flask 4/10, jar 7/10, tumbler 5/10, wineglass 2/10
- background: 0 21/50
- lighting: 0 21/50
- distractors: 1 9/16, 2 7/19, 3 5/15
- look-alike: 0 21/50
- thermal crossover: 0 19/44, 1 2/6
- liquid: empty 21/50
- depth see-through: 0 16/39, 1 5/11

## L0_distract_a03_rgb_500demos

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.32 (16/50) | 0.21-0.46 | 0.46 | 0.02 | 0.08 | 0.12 | 0.00 | 0.00 |
| g_heldout_all | 0.34 (17/50) | 0.22-0.48 | 0.44 | 0.06 | 0.08 | 0.06 | 0.02 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.04, ep50 0.22, ep75 0.26, ep100 0.38
val loss: lowest 0.0023 at epoch 86, last 0.0031 at epoch 100; train loss last 0.0014

success by condition on a_in_distribution (successes/episodes):
- object: bottle 4/10, flask 0/10, jar 7/10, tumbler 5/10, wineglass 0/10
- background: 0 16/50
- lighting: 0 16/50
- distractors: 1 5/16, 2 6/19, 3 5/15
- look-alike: 0 16/50
- thermal crossover: 0 14/44, 1 2/6
- liquid: empty 16/50
- depth see-through: 0 16/50

## L0_distract_a04_rgb_500demos

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.48 (24/50) | 0.35-0.61 | 0.36 | 0.02 | 0.08 | 0.06 | 0.00 | 0.00 |
| g_heldout_all | 0.48 (24/50) | 0.35-0.61 | 0.32 | 0.06 | 0.10 | 0.04 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.20, ep50 0.26, ep75 0.52, ep100 0.46
val loss: lowest 0.0023 at epoch 86, last 0.0028 at epoch 100; train loss last 0.0014

success by condition on a_in_distribution (successes/episodes):
- object: bottle 3/10, flask 2/10, jar 9/10, tumbler 7/10, wineglass 3/10
- background: 0 24/50
- lighting: 0 24/50
- distractors: 1 8/16, 2 9/19, 3 7/15
- look-alike: 0 24/50
- thermal crossover: 0 21/44, 1 3/6
- liquid: empty 24/50
- depth see-through: 0 24/50

## L0_distract_s50_rgbd_500demos

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.42 (21/50) | 0.29-0.56 | 0.38 | 0.04 | 0.08 | 0.08 | 0.00 | 0.00 |
| g_heldout_all | 0.62 (31/50) | 0.48-0.74 | 0.20 | 0.08 | 0.08 | 0.02 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.08, ep50 0.26, ep75 0.54, ep100 0.46
val loss: lowest 0.0029 at epoch 90, last 0.0035 at epoch 100; train loss last 0.0019

success by condition on a_in_distribution (successes/episodes):
- object: bottle 2/10, flask 2/10, jar 9/10, tumbler 5/10, wineglass 3/10
- background: 0 21/50
- lighting: 0 21/50
- distractors: 1 9/16, 2 7/19, 3 5/15
- look-alike: 0 21/50
- thermal crossover: 0 19/44, 1 2/6
- liquid: empty 21/50
- depth see-through: 0 15/28, 1 6/22

## L0_distract_s50_rgbdt_500demos

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.60 (30/50) | 0.46-0.72 | 0.24 | 0.04 | 0.08 | 0.02 | 0.02 | 0.00 |
| g_heldout_all | 0.56 (28/50) | 0.42-0.69 | 0.28 | 0.08 | 0.08 | 0.00 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.22, ep50 0.38, ep75 0.56, ep100 0.58
val loss: lowest 0.0043 at epoch 68, last 0.0057 at epoch 100; train loss last 0.0013

success by condition on a_in_distribution (successes/episodes):
- object: bottle 4/10, flask 6/10, jar 9/10, tumbler 5/10, wineglass 6/10
- background: 0 30/50
- lighting: 0 30/50
- distractors: 1 10/16, 2 11/19, 3 9/15
- look-alike: 0 30/50
- thermal crossover: 0 30/44, 1 0/6
- liquid: empty 30/50
- depth see-through: 0 17/28, 1 13/22
