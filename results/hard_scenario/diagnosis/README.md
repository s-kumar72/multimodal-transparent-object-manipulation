# Why does RGBDT fall on the hard levels? (plan, written before the runs)

**Where we are.** With 200 demos and 101 epochs, RGBDT reached 0.92 on L0, 0.60 on L1, 0.50 on L2, 0.10 on L3b and
0.14 on L4b (test set (a), 50 episodes); RGB on L4b reached 0.00. L1 with 200 demos for 201 epochs stayed at 0.56.
**Goal:** a level where RGB is low (≤ 0.3-0.4 is fine) and RGBDT reaches 0.8-0.9.

**Runs.** One change from the 200-demo runs each; everything else as in `../sweep/` (101 epochs, rollouts on (a)
every 25 epochs, the last checkpoint on test sets (a) and (g), 50 episodes each). Variant rgbdt unless noted.
"ImageNet" = `policy.obs_encoder.rgb_model.weights=IMAGENET1K_V1`: every image encoder (rgb, depth, thermal) starts
from ImageNet ResNet-18 weights instead of random ones; the architecture is unchanged.

| id | level | demos | change | question | result folder |
|---|---|---|---|---|---|
| D2 | L4b | 500 | more demos | is the main scenario a data problem? | `L4b_rgbdt_500demos` |
| D3 | L4b (rgbd) | 500 | more demos | with more data, does thermal add over rgbd? | `L4b_rgbd_500demos` |
| P4 | L4b | 200 | ImageNet | is it the randomly initialized encoders? | `L4b_rgbdt_200demos_imagenet` |
| 1a | L0_faint | 200 | L0 + only the faint glass | does the faint glass alone cause the L0 -> L1 drop? | `L0_faint_rgbdt_200demos` |
| 1b | L1_alpha02 | 200 | L1 without the faint glass | does L1 recover without it? | `L1_alpha02_rgbdt_200demos` |
| 1c | L1 | 500 | more demos | is the L1 drop a data problem? | `L1_rgbdt_500demos` |
| 1d | L1 | 200 | ImageNet | is the L1 drop an encoder problem? | `L1_rgbdt_200demos_imagenet` |
| 2a | L3b | 500 | more demos | is the L2 -> L3b drop a data problem? | `L3b_rgbdt_500demos` |
| 2a' | L3b | 200 | ImageNet | ... an encoder problem? | `L3b_rgbdt_200demos_imagenet` |
| 2b | L3b_mild | 200 | depth defects: missing 0.1-0.5, wrong depth 0-10% | is it how strong the defects are? | `L3b_mild_rgbdt_200demos` |

The new levels are `sim/configs/L0_faint.yaml`, `L1_alpha02.yaml` and `L3b_mild.yaml`. 500 demos x 101 epochs is
about 41k steps, the budget at which the 500-demo main-scene run reached 0.96; L1 with 200 demos for 201 epochs
already had about 34k steps, so a gain at 500 demos comes from the data. 500-demo sets use seeds 0-499 and contain
the 200-demo sets.

**Baselines (200 demos, rgbdt, test set (a)):** L0 0.92, L1 0.60, L2 0.50, L3b 0.10, L4b 0.14 (`../sweep/L4b_rgbdt`);
L4b rgbd 0.08 (`../sweep/L4b_rgbd`).

**Reading the results.** With 50 episodes, differences under about 0.15 are within noise; use
`python policy/summarize_runs.py compare <baseline run> <new run>` (Fisher exact test).

| finding | meaning |
|---|---|
| 1a low, 1b near 0.9 | the faint glass causes the L1 drop |
| 1a near 0.9, 1b low | the rest of the shared scene (distractors, tables, light, liquid, thermal) causes it |
| 1c or 1d well above 0.60 | the L1 drop is (also) data / encoder |
| 2a or 2a' well above 0.10 | the L3b drop is data / encoder; 2b high: the defects are too strong |
| D2 or P4 near 0.8 | L4b becomes usable with that change; then rgb / rgbd / rgbdt and L0 with it |

Whatever the results, all ten runs are reported.

**Run:** `bash scripts/run_diagnosis.sh` (four GPUs; `scripts/run_experiment.sh` for a single run). Results land in
`results/hard_scenario/diagnosis/<result folder>/`.
