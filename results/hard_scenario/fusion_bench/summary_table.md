
## L0_bench_rgbdt_500demos_cuegate

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.56 (28/50) | 0.42-0.69 | 0.30 | 0.00 | 0.10 | 0.04 | 0.00 | 0.00 |
| g_heldout_all | 0.54 (27/50) | 0.40-0.67 | 0.28 | 0.04 | 0.12 | 0.02 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.42, ep50 0.48, ep75 0.58, ep100 0.58
val loss: lowest 0.0072 at epoch 74, last 0.0090 at epoch 100; train loss last 0.0024

success by condition on a_in_distribution (successes/episodes):
- object: bottle 3/10, flask 4/10, jar 10/10, tumbler 6/10, wineglass 5/10
- background: 0 28/50
- lighting: 0 28/50
- distractors: 1 11/16, 2 10/19, 3 7/15
- look-alike: 0 28/50
- thermal crossover: 0 26/44, 1 2/6
- liquid: empty 28/50
- depth see-through: 0 19/39, 1 9/11

## L0_bench_rgbdt_500demos_cuegate_seed43

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.68 (34/50) | 0.54-0.79 | 0.20 | 0.00 | 0.08 | 0.04 | 0.00 | 0.00 |
| g_heldout_all | 0.60 (30/50) | 0.46-0.72 | 0.22 | 0.04 | 0.12 | 0.02 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.46, ep50 0.58, ep75 0.58, ep100 0.62
val loss: lowest 0.0072 at epoch 68, last 0.0092 at epoch 100; train loss last 0.0024

success by condition on a_in_distribution (successes/episodes):
- object: bottle 4/10, flask 8/10, jar 8/10, tumbler 8/10, wineglass 6/10
- background: 0 34/50
- lighting: 0 34/50
- distractors: 1 12/16, 2 12/19, 3 10/15
- look-alike: 0 34/50
- thermal crossover: 0 32/44, 1 2/6
- liquid: empty 34/50
- depth see-through: 0 25/39, 1 9/11

## L0_bench_rgbdt_500demos_fusion

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.50 (25/50) | 0.37-0.63 | 0.30 | 0.00 | 0.08 | 0.10 | 0.02 | 0.00 |
| g_heldout_all | 0.62 (31/50) | 0.48-0.74 | 0.14 | 0.00 | 0.18 | 0.04 | 0.02 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.38, ep50 0.56, ep75 0.54, ep100 0.60
val loss: lowest 0.0030 at epoch 77, last 0.0044 at epoch 104; train loss last 0.0011

success by condition on a_in_distribution (successes/episodes):
- object: bottle 4/10, flask 7/10, jar 5/10, tumbler 5/10, wineglass 4/10
- background: 0 25/50
- lighting: 0 25/50
- distractors: 1 6/16, 2 10/19, 3 9/15
- look-alike: 0 25/50
- thermal crossover: 0 21/44, 1 4/6
- liquid: empty 25/50
- depth see-through: 0 22/39, 1 3/11

## L0_bench_rgbdt_500demos_fusion_seed43

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.36 (18/50) | 0.24-0.50 | 0.46 | 0.02 | 0.08 | 0.06 | 0.02 | 0.00 |
| g_heldout_all | 0.50 (25/50) | 0.37-0.63 | 0.34 | 0.00 | 0.10 | 0.04 | 0.02 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.16, ep50 0.28, ep75 0.44, ep100 0.34
val loss: lowest 0.0028 at epoch 89, last 0.0035 at epoch 100; train loss last 0.0014

success by condition on a_in_distribution (successes/episodes):
- object: bottle 2/10, flask 1/10, jar 7/10, tumbler 6/10, wineglass 2/10
- background: 0 18/50
- lighting: 0 18/50
- distractors: 1 7/16, 2 5/19, 3 6/15
- look-alike: 0 18/50
- thermal crossover: 0 16/44, 1 2/6
- liquid: empty 18/50
- depth see-through: 0 14/39, 1 4/11

## L0_bench_rgbdt_500demos_gate_cue2

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.54 (27/50) | 0.40-0.67 | 0.38 | 0.00 | 0.06 | 0.02 | 0.00 | 0.00 |
| g_heldout_all | 0.54 (27/50) | 0.40-0.67 | 0.28 | 0.06 | 0.08 | 0.04 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.56, ep50 0.56, ep75 0.56, ep100 0.50
val loss: lowest 0.0074 at epoch 74, last 0.0093 at epoch 100; train loss last 0.0023

success by condition on a_in_distribution (successes/episodes):
- object: bottle 2/10, flask 5/10, jar 7/10, tumbler 7/10, wineglass 6/10
- background: 0 27/50
- lighting: 0 27/50
- distractors: 1 12/16, 2 6/19, 3 9/15
- look-alike: 0 27/50
- thermal crossover: 0 26/44, 1 1/6
- liquid: empty 27/50
- depth see-through: 0 21/39, 1 6/11

## L0_bench_rgbdt_500demos_gate_nohole

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.64 (32/50) | 0.50-0.76 | 0.12 | 0.00 | 0.12 | 0.12 | 0.00 | 0.00 |
| g_heldout_all | 0.60 (30/50) | 0.46-0.72 | 0.22 | 0.04 | 0.10 | 0.04 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.02, ep25 0.60, ep50 0.62, ep75 0.68, ep100 0.62
val loss: lowest 0.0075 at epoch 71, last 0.0088 at epoch 100; train loss last 0.0017

success by condition on a_in_distribution (successes/episodes):
- object: bottle 3/10, flask 6/10, jar 9/10, tumbler 7/10, wineglass 7/10
- background: 0 32/50
- lighting: 0 32/50
- distractors: 1 9/16, 2 10/19, 3 13/15
- look-alike: 0 32/50
- thermal crossover: 0 29/44, 1 3/6
- liquid: empty 32/50
- depth see-through: 0 27/39, 1 5/11

## L0_bench_rgbdt_500demos_static

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.56 (28/50) | 0.42-0.69 | 0.30 | 0.04 | 0.08 | 0.00 | 0.02 | 0.00 |
| g_heldout_all | 0.58 (29/50) | 0.44-0.71 | 0.26 | 0.04 | 0.08 | 0.02 | 0.02 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.32, ep50 0.46, ep75 0.50, ep100 0.48
val loss: lowest 0.0059 at epoch 62, last 0.0082 at epoch 100; train loss last 0.0014

success by condition on a_in_distribution (successes/episodes):
- object: bottle 7/10, flask 6/10, jar 7/10, tumbler 3/10, wineglass 5/10
- background: 0 28/50
- lighting: 0 28/50
- distractors: 1 11/16, 2 8/19, 3 9/15
- look-alike: 0 28/50
- thermal crossover: 0 25/44, 1 3/6
- liquid: empty 28/50
- depth see-through: 0 22/39, 1 6/11

## L0_bench_rgbdt_500demos_static_cue2

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.52 (26/50) | 0.39-0.65 | 0.30 | 0.02 | 0.14 | 0.02 | 0.00 | 0.00 |
| g_heldout_all | 0.56 (28/50) | 0.42-0.69 | 0.28 | 0.02 | 0.10 | 0.02 | 0.02 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.38, ep50 0.48, ep75 0.52, ep100 0.58
val loss: lowest 0.0054 at epoch 66, last 0.0076 at epoch 100; train loss last 0.0014

success by condition on a_in_distribution (successes/episodes):
- object: bottle 5/10, flask 5/10, jar 7/10, tumbler 6/10, wineglass 3/10
- background: 0 26/50
- lighting: 0 26/50
- distractors: 1 8/16, 2 10/19, 3 8/15
- look-alike: 0 26/50
- thermal crossover: 0 25/44, 1 1/6
- liquid: empty 26/50
- depth see-through: 0 19/39, 1 7/11

## L0_bench_rgbdt_500demos_gate_nohole_seed43

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.72 (36/50) | 0.58-0.83 | 0.08 | 0.00 | 0.14 | 0.06 | 0.00 | 0.00 |
| g_heldout_all | 0.60 (30/50) | 0.46-0.72 | 0.18 | 0.02 | 0.16 | 0.02 | 0.02 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.62, ep50 0.64, ep75 0.66, ep100 0.76
val loss: lowest 0.0073 at epoch 57, last 0.0102 at epoch 100; train loss last 0.0017

success by condition on a_in_distribution (successes/episodes):
- object: bottle 5/10, flask 6/10, jar 10/10, tumbler 9/10, wineglass 6/10
- background: 0 36/50
- lighting: 0 36/50
- distractors: 1 12/16, 2 13/19, 3 11/15
- look-alike: 0 36/50
- thermal crossover: 0 34/44, 1 2/6
- liquid: empty 36/50
- depth see-through: 0 28/39, 1 8/11

## L0_bench_rgbdt_500demos_gate_vis

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.70 (35/50) | 0.56-0.81 | 0.10 | 0.02 | 0.04 | 0.14 | 0.00 | 0.00 |
| g_heldout_all | 0.68 (34/50) | 0.54-0.79 | 0.10 | 0.02 | 0.14 | 0.04 | 0.02 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.54, ep50 0.70, ep75 0.76, ep100 0.70
val loss: lowest 0.0083 at epoch 81, last 0.0094 at epoch 100; train loss last 0.0024

success by condition on a_in_distribution (successes/episodes):
- object: bottle 4/10, flask 6/10, jar 10/10, tumbler 8/10, wineglass 7/10
- background: 0 35/50
- lighting: 0 35/50
- distractors: 1 11/16, 2 13/19, 3 11/15
- look-alike: 0 35/50
- thermal crossover: 0 31/44, 1 4/6
- liquid: empty 35/50
- depth see-through: 0 27/39, 1 8/11

## L0_bench_rgbdt_500demos_static_nohole

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.58 (29/50) | 0.44-0.71 | 0.22 | 0.04 | 0.16 | 0.00 | 0.00 | 0.00 |
| g_heldout_all | 0.56 (28/50) | 0.42-0.69 | 0.28 | 0.02 | 0.10 | 0.02 | 0.02 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.38, ep50 0.54, ep75 0.62, ep100 0.56
val loss: lowest 0.0057 at epoch 63, last 0.0078 at epoch 100; train loss last 0.0013

success by condition on a_in_distribution (successes/episodes):
- object: bottle 5/10, flask 6/10, jar 9/10, tumbler 4/10, wineglass 5/10
- background: 0 29/50
- lighting: 0 29/50
- distractors: 1 10/16, 2 8/19, 3 11/15
- look-alike: 0 29/50
- thermal crossover: 0 26/44, 1 3/6
- liquid: empty 29/50
- depth see-through: 0 23/39, 1 6/11

## L0_bench_rgbdt_500demos_vis_route

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.72 (36/50) | 0.58-0.83 | 0.14 | 0.00 | 0.08 | 0.06 | 0.00 | 0.00 |
| g_heldout_all | 0.66 (33/50) | 0.52-0.78 | 0.16 | 0.04 | 0.10 | 0.04 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.02, ep25 0.62, ep50 0.66, ep75 0.66, ep100 0.72
val loss: lowest 0.0080 at epoch 75, last 0.0098 at epoch 100; train loss last 0.0023

success by condition on a_in_distribution (successes/episodes):
- object: bottle 5/10, flask 8/10, jar 10/10, tumbler 7/10, wineglass 6/10
- background: 0 36/50
- lighting: 0 36/50
- distractors: 1 12/16, 2 11/19, 3 13/15
- look-alike: 0 36/50
- thermal crossover: 0 31/44, 1 5/6
- liquid: empty 36/50
- depth see-through: 0 28/39, 1 8/11

## L0_bench_rgbdt_500demos_vis_route_seed43

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.76 (38/50) | 0.63-0.86 | 0.06 | 0.02 | 0.08 | 0.08 | 0.00 | 0.00 |
| g_heldout_all | 0.66 (33/50) | 0.52-0.78 | 0.16 | 0.04 | 0.12 | 0.02 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.50, ep50 0.66, ep75 0.74, ep100 0.68
val loss: lowest 0.0087 at epoch 59, last 0.0131 at epoch 100; train loss last 0.0023

success by condition on a_in_distribution (successes/episodes):
- object: bottle 6/10, flask 7/10, jar 10/10, tumbler 9/10, wineglass 6/10
- background: 0 38/50
- lighting: 0 38/50
- distractors: 1 13/16, 2 13/19, 3 12/15
- look-alike: 0 38/50
- thermal crossover: 0 33/44, 1 5/6
- liquid: empty 38/50
- depth see-through: 0 30/39, 1 8/11

## L0_bench_rgbdt_500demos_vis_soft

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.68 (34/50) | 0.54-0.79 | 0.18 | 0.00 | 0.08 | 0.06 | 0.00 | 0.00 |
| g_heldout_all | 0.66 (33/50) | 0.52-0.78 | 0.14 | 0.02 | 0.14 | 0.02 | 0.02 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.56, ep50 0.60, ep75 0.68, ep100 0.66
val loss: lowest 0.0090 at epoch 58, last 0.0114 at epoch 100; train loss last 0.0022

success by condition on a_in_distribution (successes/episodes):
- object: bottle 6/10, flask 8/10, jar 10/10, tumbler 6/10, wineglass 4/10
- background: 0 34/50
- lighting: 0 34/50
- distractors: 1 11/16, 2 11/19, 3 12/15
- look-alike: 0 34/50
- thermal crossover: 0 30/44, 1 4/6
- liquid: empty 34/50
- depth see-through: 0 27/39, 1 7/11

## L0_bench_rgbdt_500demos_nogate_rgbonly

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.68 (34/50) | 0.54-0.79 | 0.20 | 0.00 | 0.08 | 0.04 | 0.00 | 0.00 |
| g_heldout_all | 0.56 (28/50) | 0.42-0.69 | 0.14 | 0.04 | 0.16 | 0.10 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.44, ep50 0.62, ep75 0.64, ep100 0.62
val loss: lowest 0.0090 at epoch 61, last 0.0117 at epoch 100; train loss last 0.0023

success by condition on a_in_distribution (successes/episodes):
- object: bottle 6/10, flask 5/10, jar 10/10, tumbler 7/10, wineglass 6/10
- background: 0 34/50
- lighting: 0 34/50
- distractors: 1 13/16, 2 12/19, 3 9/15
- look-alike: 0 34/50
- thermal crossover: 0 29/44, 1 5/6
- liquid: empty 34/50
- depth see-through: 0 26/39, 1 8/11

## L0_bench_rgbdt_500demos_vis_route_seed44

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.76 (38/50) | 0.63-0.86 | 0.12 | 0.00 | 0.10 | 0.02 | 0.00 | 0.00 |
| g_heldout_all | 0.60 (30/50) | 0.46-0.72 | 0.22 | 0.04 | 0.06 | 0.08 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.52, ep50 0.64, ep75 0.78, ep100 0.76
val loss: lowest 0.0073 at epoch 63, last 0.0096 at epoch 100; train loss last 0.0024

success by condition on a_in_distribution (successes/episodes):
- object: bottle 5/10, flask 7/10, jar 10/10, tumbler 8/10, wineglass 8/10
- background: 0 38/50
- lighting: 0 38/50
- distractors: 1 14/16, 2 11/19, 3 13/15
- look-alike: 0 38/50
- thermal crossover: 0 34/44, 1 4/6
- liquid: empty 38/50
- depth see-through: 0 30/39, 1 8/11

## L0_bench_rgbdt_500demos_nogate_rgbonly_seed43

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.64 (32/50) | 0.50-0.76 | 0.20 | 0.00 | 0.06 | 0.10 | 0.00 | 0.00 |
| g_heldout_all | 0.58 (29/50) | 0.44-0.71 | 0.24 | 0.06 | 0.08 | 0.04 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.50, ep50 0.66, ep75 0.64, ep100 0.60
val loss: lowest 0.0101 at epoch 60, last 0.0145 at epoch 100; train loss last 0.0022

success by condition on a_in_distribution (successes/episodes):
- object: bottle 6/10, flask 4/10, jar 10/10, tumbler 6/10, wineglass 6/10
- background: 0 32/50
- lighting: 0 32/50
- distractors: 1 10/16, 2 12/19, 3 10/15
- look-alike: 0 32/50
- thermal crossover: 0 28/44, 1 4/6
- liquid: empty 32/50
- depth see-through: 0 26/39, 1 6/11

## L0_bench_rgbdt_500demos_nogate_rgbonly_seed44

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.62 (31/50) | 0.48-0.74 | 0.18 | 0.00 | 0.16 | 0.04 | 0.00 | 0.00 |
| g_heldout_all | 0.62 (31/50) | 0.48-0.74 | 0.24 | 0.04 | 0.08 | 0.02 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.56, ep50 0.62, ep75 0.66, ep100 0.70
val loss: lowest 0.0090 at epoch 47, last 0.0113 at epoch 100; train loss last 0.0023

success by condition on a_in_distribution (successes/episodes):
- object: bottle 3/10, flask 6/10, jar 9/10, tumbler 7/10, wineglass 6/10
- background: 0 31/50
- lighting: 0 31/50
- distractors: 1 10/16, 2 11/19, 3 10/15
- look-alike: 0 31/50
- thermal crossover: 0 28/44, 1 3/6
- liquid: empty 31/50
- depth see-through: 0 26/39, 1 5/11

## L0_bench_rgbdt_500demos_nogate_allon

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.56 (28/50) | 0.42-0.69 | 0.20 | 0.00 | 0.18 | 0.06 | 0.00 | 0.00 |
| g_heldout_all | 0.56 (28/50) | 0.42-0.69 | 0.24 | 0.00 | 0.16 | 0.02 | 0.02 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.46, ep50 0.54, ep75 0.60, ep100 0.52
val loss: lowest 0.0095 at epoch 48, last 0.0141 at epoch 100; train loss last 0.0022

success by condition on a_in_distribution (successes/episodes):
- object: bottle 5/10, flask 4/10, jar 9/10, tumbler 6/10, wineglass 4/10
- background: 0 28/50
- lighting: 0 28/50
- distractors: 1 10/16, 2 9/19, 3 9/15
- look-alike: 0 28/50
- thermal crossover: 0 26/44, 1 2/6
- liquid: empty 28/50
- depth see-through: 0 21/39, 1 7/11

## L0_bench_rgbd_500demos_seed44

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.62 (31/50) | 0.48-0.74 | 0.22 | 0.06 | 0.08 | 0.02 | 0.00 | 0.00 |
| g_heldout_all | 0.44 (22/50) | 0.31-0.58 | 0.22 | 0.14 | 0.10 | 0.08 | 0.02 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.26, ep50 0.32, ep75 0.36, ep100 0.62
val loss: lowest 0.0023 at epoch 94, last 0.0025 at epoch 100; train loss last 0.0021

success by condition on a_in_distribution (successes/episodes):
- object: bottle 7/10, flask 6/10, jar 7/10, tumbler 7/10, wineglass 4/10
- background: 0 31/50
- lighting: 0 31/50
- distractors: 1 9/16, 2 12/19, 3 10/15
- look-alike: 0 31/50
- thermal crossover: 0 25/44, 1 6/6
- liquid: empty 31/50
- depth see-through: 0 30/39, 1 1/11

## L0_bench_rgbdt_500demos_seed44

| test set | success | 95% CI | missed_grasp | other | dropped | wrong_height | wrong_object | timeout |
|---|---|---|---|---|---|---|---|---|
| a_in_distribution | 0.70 (35/50) | 0.56-0.81 | 0.18 | 0.04 | 0.06 | 0.02 | 0.00 | 0.00 |
| g_heldout_all | 0.60 (30/50) | 0.46-0.72 | 0.20 | 0.14 | 0.06 | 0.00 | 0.00 | 0.00 |

rollouts on (a) during training: ep0 0.00, ep25 0.32, ep50 0.40, ep75 0.62, ep100 0.72
val loss: lowest 0.0034 at epoch 84, last 0.0038 at epoch 100; train loss last 0.0014

success by condition on a_in_distribution (successes/episodes):
- object: bottle 5/10, flask 6/10, jar 9/10, tumbler 8/10, wineglass 7/10
- background: 0 35/50
- lighting: 0 35/50
- distractors: 1 10/16, 2 13/19, 3 12/15
- look-alike: 0 35/50
- thermal crossover: 0 32/44, 1 3/6
- liquid: empty 35/50
- depth see-through: 0 29/39, 1 6/11
