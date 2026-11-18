# Ablations of the learned attention weighting

The first round on L0_bench (`results/hard_scenario/fusion_bench/README.md`) left the fusion policy below the
static-fusion baseline: 52.8% against 62.6% over 800 episodes. There were three clear symptoms:

- **Seed spread.** Seed 42 was level with the baseline (61.8%), seed 43 far below it (43.8%; p = 5e-7 between the
  two seeds).
- **Imprecise grasps.** Most failures are missed grasps, with the gripper a median 2.5 cm from the glass. The
  policy finds the glass but closes in the wrong place.
- **A mixed sensor pattern.** See-through improves over the baseline (37.8% vs 21.1%), but normal episodes drop
  (57.1% vs 74.7%). Crossover is about level (51.2% vs 54.8%).

The segmentation network on its own behaves as intended: 87% IoU, and the thermal weight falls from 0.29 to 0.05 on
crossover frames. So the open questions are mostly about what happens to it inside the policy. The ablations below
are designed to answer these one at a time.

The `static`, `gate_nohole` and `gate_cue2` runs of the first round were ablations of cue gating, not of this
method. Only two things have been run for learned attention weighting so far: the two seeds of the full model.

![Where each ablation acts](ablations/fig_architecture.png)

## The ablations

Blue tags (S) change the segmentation pre-training (`fusion/train_seg.py`, a fraction of a policy run). Orange tags (P)
change the policy training (about 5 h per run on one GPU, from the first round). Every P run is trained with seeds 42 and 43 by default. One seed can't be trusted here as the full model's two seeds differ by 18 points.

### QUESTION 1: Does pre-training help, and does policy training keep it?

The encoder starts from the segmentation network (`mix_ws`) and is then trained further, with the policy's action
loss only. Nothing tells it to keep segmenting glass or to keep its sensor weights. If that drifts in a different
direction for each seed, it would explain both the seed spread and the weaker normal episodes.

| id | change | what it tells us |
|---|---|---|
| P1 | no pre-training: the same encoder trained from scratch with the policy | Whether pre-training helps at all. If P1 is as good as P0, the segmentation stage is not paying off. |
| P2 | pre-trained encoder frozen: only the readout and the U-Net learn | Whether fine-tuning hurts. If P2 beats P0 and its seeds agree, drift during fine-tuning was the problem. |
| P3 | frozen for the first 10k steps, then trained | A middle ground: the U-Net settles before the encoder may change. |
| P4 | trained, but anchored to a frozen copy of the pre-trained network (its glass mask and sensor weights) | Keeps what pre-training learned while still adapting. If P4 beats both P0 and P2, adaptation helps but drift hurts. |

### QUESTION 2: Do the sensor weights matter?

| id | change | what it tells us |
|---|---|---|
| P5 | weights fixed at 1/3 each inside our encoder | The direct test of the gate. If P5 scores like P0, the weights do nothing for the policy. If P0 is clearly better on see-through or crossover, the weights are doing their job there. |
| P6 | weight supervision from the episode flags: depth pushed down in see-through episodes, thermal in crossover episodes, faded out over 20k steps | Gives the weights a reliability signal during policy training too. The cue-gating version for depth (`gate_cue2`) did not form a depth gate. The training log reports `w_depth_st` vs `w_depth_notst` and `w_thermal_xo` vs `w_thermal_notxo`, so it can be checked early. |
| P7 | thermal contrast removed in half of the training samples (`policy/glass_dataset.py` corruption, depth holes off) | Teaches the policy itself to cope with flat thermal. Depth holes stay off, because in cue gating they kept the depth weight flat (`gate_nohole`). |

### QUESTION 3: What the policy reads from the encoder (grasp precision)

![The readout](ablations/fig_readout.png)

The policy gets a few summaries of the fused features, and the glass position is computed on the 6 × 6 fusion grid.
The 2.5 cm misses suggest that position is too coarse to grasp with.

| id | change | what it tells us |
|---|---|---|
| P8 | only the image average and the sensor weights, no glass-specific parts | Whether the segmentation-based readout helps at all. If P8 is as good as P0, the mask features are unused. |
| P9 | adds the glass position, spread and peak at full 84 px resolution, and 32 learned keypoints (spatial softmax, as in robomimic) | Whether precision is the bottleneck. The thing to watch is the miss distance column, not only success. |

### Sensors and pre-training data

| id | change | what it tells us |
|---|---|---|
| P10 | our encoder with RGB-D only (thermal treated as missing) | Whether thermal helps or hurts inside our model. Prior work warns that adding sensors can lower success; this shows whether that happens here. |
| P11 | encoder initialized from S4 (below) instead of `mix_ws` | Whether a see-through-aware segmenter carries over to the policy. |

### Segmentation pre-training (S)

![The gate and its ablations](ablations/fig_gate.png)

In cue gating, the depth-hole corruption kept the depth weight flat: the gate learned "bad depth = holes", while
see-through depth has no holes and is simply wrong. Our pre-training uses the same hole corruption. These runs test
that, and try a corruption that matches the real failure.

| id | run name | change | what it tells us |
|---|---|---|---|
| S0 | `mix_ws` | the reference, retrained on the new sim frames | Comparison point (new frames are needed for S3/S4). |
| S1 | `mix_ws_nohole` | no depth-hole and wrong-depth corruption | Does the depth weight on the glass drop in see-through frames once the hole corruption is gone? |
| S2 | `mix_ws_stx` | holes replaced by a see-through corruption: depth on the glass set to the surrounding table depth | A corruption shaped like the real failure. |
| S3 | `mix_ws_supst` | the gate loss also says "depth is unreliable on the glass" in see-through frames | A direct signal for depth, like crossover already is for thermal. |
| S4 | `mix_ws_stx_supst` | S2 + S3 | The candidate for P11. |
| S5 | `mix_ws_nocues` | no reliability cues | Do the cues carry the weights, or do the features? |
| S6 | `mix_ws_nosup` | no gate loss | Without it, the weights collapsed in the short CPU test (`README.md`). Does that hold at full scale? |

The S runs are judged on segmentation, not on grasping. The summary (`results/fusion/summary.md`, last table) gives
per-image IoU on normal / see-through / crossover frames. It also gives the depth weight on the glass in normal vs
see-through frames, and an AUC for how well that weight separates the two (0.5 = not at all; `gate_nohole` reached
0.95 in cue gating).

### What each result would mean

| if… | then… |
|---|---|
| P2 or P4 beats P0 and its two seeds agree | fine-tuning drift caused the spread; use the frozen / anchored encoder |
| P1 ≈ P0 | pre-training is not what helps; the gain (if any) is the architecture |
| P5 ≈ P0 everywhere | the learned weights do not matter for the policy (worth reporting) |
| P0 > P5 on see-through or crossover only | the weights help exactly where a sensor fails, the claim of the method |
| P9 lowers the miss distance and raises normal | grasp precision was the bottleneck |
| P10 ≥ P0 | thermal does not help inside our encoder (as prior work warns) |
| S1/S2 raise the depth-weight AUC above ~0.8 | the hole corruption was blocking the depth weight, as in cue gating |

## Running them

Everything is in `fusion/ablations/run_ablations.sh`. It uses `scripts/run_experiment.sh`, the same protocol as the
first round (500 demos, 101 epochs, last checkpoint, test sets a and g). After training, each run is re-evaluated
with 200 episodes per test set, with the sensor weights saved at every step (`--save_w`). That gives 800 episodes
per row over two seeds, matching the first round.

| tier | runs | count | why first |
|---|---|---|---|
| 1 | P0 (3rd seed), P2, P4, P5, P9 | 9 | the main questions: seed spread, drift, the gate, precision |
| 2 | P1, P6, P7, P10 | 8 | pre-training, weight supervision, corruption, thermal |
| 3 | P3, P8, P11 | 6 | P11 needs S4 first |

```bash
# segmentation runs first (new sim frames with the see-through flag; not timed yet)
DATA=data/fusion_v2 SIM_LEVELS=L0_bench \
  RUNS="mix_ws mix_ws_nohole mix_ws_stx mix_ws_supst mix_ws_stx_supst mix_ws_nocues mix_ws_nosup" bash fusion/train.sh

# policy runs, tier by tier; DRY=1 prints the commands without running anything
DRY=1 TIER=1 bash fusion/ablations/run_ablations.sh
TIER=1 PER_GPU=2 bash fusion/ablations/run_ablations.sh
TIER=2 PER_GPU=2 bash fusion/ablations/run_ablations.sh
TIER=3 PER_GPU=2 bash fusion/ablations/run_ablations.sh      # after S4 exists
```

`SEG_CKPT` points at the `mix_ws` checkpoint used in the first round (default `data/fusion/runs/mix_ws/best.pt`),
so P0–P10 start from the same network as before. `PER_GPU=2` runs two trainings per GPU, which fits on an A6000;
the first round's RAM problems came from running six trainings and three evaluations at once.

Single runs work the same way as before, for example P4 with seed 43:

```bash
GPU=0 TAG=fusion_anchor_seed43 TRAIN=scripts/train_fusion_policy.sh GROUP=fusion_ablations EVAL_EXTRA=--save_w \
  EXTRA="training.seed=43 policy.obs_encoder.pretrained=data/fusion/runs/mix_ws/best.pt policy.obs_encoder.anchor=true policy.anchor_weight=1.0" \
  bash scripts/run_experiment.sh L0_bench rgbdt 500
```

The two first-round fusion checkpoints were trained before the weight logging existed. They can be re-evaluated with
it through `fusion/eval_fusion.py`, which takes the same options as `policy/eval.py`:

```bash
python -m fusion.eval_fusion -c data/runs/L0_bench_rgbdt_500demos_fusion/checkpoints/latest.ckpt \
  -o results/hard_scenario/fusion_ablations/n200/L0_bench_rgbdt_500demos_fusion --level L0_bench --n_test 200 \
  --n_envs 10 --test_set a_in_distribution --test_set g_heldout_all --save_w
```

## The results table

```bash
python -m fusion.ablation_table --spec fusion/ablations/ablations.yaml --out results/hard_scenario/fusion_ablations/table
```

This writes four files:

- `table.md`, for GitHub;
- `table.tex`, a booktabs table for report (needs `\usepackage{booktabs}`);
- `table.csv`, every number with its 95% interval;
- `table.png`, success per group with intervals.

Rows that have not been run yet show as "not run", so the table can go into the report draft now and fill in as runs
finish. The baselines and the first round already give real numbers.

How to read it:

- Each cell is the success rate in %. In brackets is the change against `rgbdt (static fusion)` on the same seeds,
  with stars for a two-sided Fisher exact test (`*` p < 0.05, `**` p < 0.01, `***` p < 0.001).
- The groups are those of the first round: normal 620, see-through 180 and crossover 84 of 800 episodes. Crossover
  overlaps the other two.
- `per seed` shows each seed's overall rate and the p-value between the best and worst seed. That is the column for
  the seed-spread question.
- `miss (cm)` is the median gripper-to-glass distance at the last close, in failed episodes. That is the column for
  grasp precision.
- `gate AUC depth / thermal` shows whether the sensor weights separate normal from see-through episodes, and crossover
  from non-crossover (0.5 = not at all). It needs `--save_w`, which the ablation runs have.
- With 12 ablations, a single p < 0.05 can happen by chance. A result is worth trusting when both seeds point the
  same way.

New rows go into `fusion/ablations/ablations.yaml`, as a name and the run folders, one per seed.

## Location of ablations in code

| ablation | switch | file |
|---|---|---|
| P1 | `policy.obs_encoder.pretrained=null` + `model_cfg` | `fusion/policy_encoder.py` |
| P2, P3 | `freeze_segmenter`, `freeze_steps` | `fusion/policy_encoder.py` |
| P4 | `obs_encoder.anchor=true`, `policy.anchor_weight` | `fusion/policy_encoder.py`, `fusion/fusion_policy.py` |
| P5 | `obs_encoder.model_overrides={gate_mode:static}` | `fusion/policy_encoder.py`, `fusion/model.py` |
| P6 | `policy.gate_sup_weight`, `gate_sup_anneal_steps` | `fusion/fusion_policy.py` |
| P7 | `task.dataset.augment={...}` | `policy/glass_dataset.py` (cue-gating branch, unchanged) |
| P8, P9 | `obs_encoder.readout=[...]`, `ssm_keypoints` | `fusion/policy_encoder.py` |
| P10 | variant `rgbd` of `scripts/train_fusion_policy.sh` | — |
| S1, S2 | `train_seg.py --aug depth_holes=0 depth_wrong=0 [depth_seethrough=0.3]` | `fusion/data.py` (`Augment`) |
| S3 | `train_seg.py --sup_seethrough` | `fusion/data.py`, `fusion/model.py` (`gate_reliability_loss`) |
| S5, S6 | `--no_cues`, `--gate_sup 0` | `fusion/model.py`, `fusion/train_seg.py` |
| table | `python -m fusion.ablation_table` | `fusion/ablation_table.py`, `fusion/ablations/ablations.yaml` |
| figures | `python fusion/ablations/make_figures.py` | `fusion/ablations/` |

All switches default to the first-round behavior, so existing checkpoints and commands work unchanged.
`fusion/collect_sim.py` now also stores each frame's `glass_seethrough` flag. Sim frames collected before this change
don't have it, which is why the S runs use a new `DATA` folder.