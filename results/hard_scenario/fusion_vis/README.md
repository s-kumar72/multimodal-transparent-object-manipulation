# Learned attention weighting on L0_vis

The fusion method of `fusion/attention/` (glass segmentation with per-sensor weights as the image encoder of Diffusion
Policy) on the visibility benchmark `L0_vis` ([`../grid_bench/README.md`](../grid_bench/README.md)). Its first round
was on `L0_bench` ([`../fusion_bench/README.md`](../fusion_bench/README.md)): 0.527 against 0.626 for the plain
policy, with misses of 2.5 cm.

## Plan (11/22, written before any run; nothing has been started)

Everything is run by `fusion/attention/run_l0_vis.sh` (`DRY=1` prints the commands):

| stage | what | output |
|---|---|---|
| frames | 96 x 96 rgb / depth / thermal frames of the 700 training scenes of `L0_vis` with the simulator's glass mask, 40 validation scenes (other seeds), 110 episodes of each test set (for reporting the segmentation only) | `data/fusion_vis/sim` |
| seg | the segmentation run `mix_ws` (these frames + the real datasets; learned per-sensor weights), as in the first round | `data/fusion_vis/runs/mix_ws`, `results/fusion/summary_fusion_vis.md` |
| policy | three policy runs started from `mix_ws`, seed 42 | `results/hard_scenario/fusion_vis/<run>/` |

| run | id in `fusion/attention/ablations.MD` | what differs |
|---|---|---|
| `L0_vis_rgbdt_700demos_fusion` | P0 | the method as it is |
| `L0_vis_rgbdt_700demos_fusion_kp` | P9 | the readout also gives the glass position at full resolution and 32 learned keypoints (the default readout has the position on a 6 x 6 grid) |
| `L0_vis_rgbdt_700demos_fusion_static` | P5 | the sensor weights are fixed at 1/3 each |

Protocol of the `L0_vis` baselines: the 700 training demos, 101 epochs, last checkpoint, default settings, 660 episodes
on each of `j_cells11` and `k_cells11_heldout`, success per cell with `policy/summarize_cells.py`.

**What these runs use that the baselines do not.** The simulator's exact glass masks of the training scenes and its
flag for the frames in which thermal does not show the glass (the gate loss of `mix_ws`), and the masks of the real
datasets; all in the segmentation pre-training only. The policy training and the evaluation see sensor images and
joint values. The runs are therefore reported as a method with extra supervision, next to the plain baselines and
not among them; P5 has the same supervision without the learned weights and is the control for them. The encoders of
the segmentation network start from ImageNet weights, as those of the strongest baseline.

**Read-out, fixed now** (one seed, so no claim about the method is made from this round):

1. Each run against the ImageNet rgbdt of seed 42 (`L0_vis_rgbdt_700demos_imagenet`) on the same episodes, per cell
   and for the macro average (sign test), with the best baseline of each cell and the reference next to it. A run
   counts as above the baseline if its macro average is above with p < 0.05; the cells in which it is below with
   p < 0.05 are listed.
2. P9 against P0, same episodes: if P9 is above with p < 0.05, the precision of the readout was a bottleneck. The
   share of episodes in which the gripper closes within 1.5 cm of the glass is reported for all three
   (`policy/analyze_failures.py`).
3. P0 against P5, same episodes, over all cells and over the nine cells in which at least one sensor does not show
   the glass fully (cells 1 to 9): if P0 is above with p < 0.05, the learned weights help; if not, they are not shown
   to matter for the policy.
4. Reported for the segmentation run: IoU on the frames of the two test sets, and the mean weight of each sensor by
   cell.

Second seeds, the frozen and anchored encoders (P2, P4) and the other ablations are decided after this round.

## Results (11/23 08:40 KST; seed 42, 1320 test episodes per run, 120 per cell)

The three policy runs and the segmentation run finished without a failed run (segmentation 38 minutes; policies
5 h 30 min to 5 h 50 min each, three at a time). All numbers below are printed by
`python fusion/attention/readout_l0_vis.py` ([`readout.txt`](readout.txt)); success per cell of a run is in its
`cells.json`.

| cell | sensors that show the glass | P0 `fusion` | P9 `fusion_kp` | P5 `fusion_static` | rgbdt ImageNet, seed 42 | best baseline (3 seeds, frozen) | reference |
|---|---|---|---|---|---|---|---|
| 0 | rgb+depth+thermal | 0.500 (below) | 0.692 | 0.283 (below) | 0.775 | 0.781 | 0.900 |
| 1 | depth+thermal | 0.333 (below) | 0.625 (below) | 0.225 (below) | 0.750 | 0.758 | 0.875 |
| 2 | rgb+thermal | 0.058 (below) | 0.550 (above) | 0.092 (below) | 0.225 | 0.397 | 0.642 |
| 3 | rgb+depth | 0.433 (below) | 0.667 | 0.225 (below) | 0.650 | 0.700 | 0.867 |
| 4 | thermal | 0.033 (below) | 0.492 (above) | 0.000 (below) | 0.192 | 0.189 | 0.608 |
| 5 | depth | 0.392 (below) | 0.583 | 0.225 (below) | 0.650 | 0.697 | 0.858 |
| 6 | rgb | 0.058 (below) | 0.492 (above) | 0.058 (below) | 0.158 | 0.256 | 0.492 |
| 7 | rgb (faint) | 0.058 | 0.367 (above) | 0.033 | 0.092 | 0.181 | 0.333 |
| 8 | depth (faint) | 0.342 (below) | 0.458 | 0.225 (below) | 0.525 | 0.486 | 0.742 |
| 9 | thermal (faint) | 0.050 | 0.342 (above) | 0.092 | 0.050 | 0.094 | 0.383 |
| 10 | all faint | 0.400 | 0.583 | 0.250 (below) | 0.483 | 0.503 | 0.717 |
| | **macro average** | 0.242 (below) | 0.532 (above) | 0.155 (below) | 0.414 | 0.458 | 0.674 |

"above" / "below": against the rgbdt baseline of seed 42 on the same episodes, sign test, p < 0.05.

**1. Against the baseline.** P9 is above it: 0.532 against 0.414 (only P9 succeeds in 319 episodes, only the
baseline in 163, p = 1.1e-12). It is above in the five cells in which depth does not show the glass (2, 4, 6, 7, 9;
+0.28 to +0.33 each) and below in cell 1 (0.625 against 0.750, p = 0.028); in the six cells in which depth shows the
glass it has 0.601 against 0.639. P0 (0.242, p = 1.6e-28) and P5 (0.155, p = 1.2e-62) are below the baseline, P0 in
cells 0 to 6 and 8, P5 in cells 0 to 6, 8 and 10. Next to the frozen numbers (three seeds against this one seed, not
tested): the macro average of P9 is above the mean of the best baseline of each cell (0.458); by cell it is higher in
2, 4, 6, 7, 9 and 10 and lower in 0, 1, 3, 5 and 8, and it is at the reference in cells 6 and 7.

**2. P9 against P0.** 0.532 against 0.242 (only P9 succeeds in 473 episodes, only P0 in 90, p = 1e-63): by the rule
above, the precision of the readout was a bottleneck. Where the gripper closes (last close of the episode):

| cells | policy | success | closes within 4 cm of the glass | within 1.5 cm | success when within 1.5 cm | median distance, cm |
|---|---|---|---|---|---|---|
| all | P0 | 0.242 | 0.560 | 0.262 | 0.720 | 3.28 |
| all | P9 | 0.532 | 0.889 | 0.623 | 0.785 | 1.11 |
| all | P5 | 0.155 | 0.450 | 0.152 | 0.721 | 4.47 |
| all | rgbdt ImageNet, seed 42 | 0.414 | 0.575 | 0.364 | 0.940 | 2.98 |
| depth shows (0, 1, 3, 5, 8, 10) | P9 | 0.601 | 0.931 | 0.706 | 0.809 | 0.92 |
| depth shows (0, 1, 3, 5, 8, 10) | rgbdt ImageNet, seed 42 | 0.639 | 0.814 | 0.599 | 0.961 | 1.03 |
| depth blind (2, 4, 6, 7, 9) | P9 | 0.448 | 0.838 | 0.523 | 0.745 | 1.44 |
| depth blind (2, 4, 6, 7, 9) | rgbdt ImageNet, seed 42 | 0.143 | 0.288 | 0.082 | 0.755 | 6.90 |

P9 finds the glass where the baseline does not (within 4 cm in 0.84 of the episodes without depth, baseline 0.29).
Once within 1.5 cm it lifts the glass less often than the baseline (0.785 against 0.940), which is why it is not
above the baseline in the cells with depth although it is within 1.5 cm more often there.

**3. P0 against P5.** P0 is above P5 over all cells (0.242 against 0.155, p = 1e-12) and over cells 1 to 9 (0.195
against 0.131, p = 1.9e-07): by the rule above, the learned weights help. What the weights are is in point 4: those
of P0 are the same in every cell, so this is one fixed weighting (thermal off) against another (1/3 each), and both
runs are below the baseline.

**4. Segmentation and weights.** `mix_ws` on the frames of the test sets: IoU 82.1 % (`j_cells11`) and 80.5 %
(`k_cells11_heldout`), most confident pixel on the glass in 100 % and 99.8 %
([`../../fusion/summary_fusion_vis.md`](../../fusion/summary_fusion_vis.md)). Its weights before the policy training:
rgb 0.43, depth 0.43, thermal 0.15; the thermal weight is 0.26 on the frames in which thermal shows the glass and
0.01 to 0.02 where it does not (AUC 0.93 and 0.96), the depth weight equals the rgb weight and does not separate the
frames in which depth sees through the glass (AUC 0.58 and 0.57). The weights of the policies during the evaluation
(rgb / depth / thermal, mean per cell): P0 0.50 / 0.50 / 0.00 in every cell, P9 0.34 / 0.34 / 0.33 in every cell
(P5 is fixed at 1/3). AUC of a sensor's weight between the cells in which it shows the glass and those in which it
does not: P0 rgb 0.53, depth 0.88 (a difference of 0.003 in the weight), thermal 0.49; P9 0.51, 0.52, 0.56. After
the policy training the weights do not follow what the sensors show: in P0 thermal is off everywhere, in P9 the
three sensors are weighted equally.

**What this round says** (one seed):

- The readout with the position at full resolution and the keypoints (P9) is the first run above the rgbdt baseline
  on `L0_vis`, by finding the glass in the cells without depth. It uses the simulator's glass masks in the
  pre-training, so it stays outside the comparison of stage 6
  ([`../grid_bench/README.md`](../grid_bench/README.md)).
- Its gain is not from weighting the sensors by what they show: its weights are equal in all cells. A control for
  that would be P9 with the weights fixed at 1/3; a control for the masks would be P9 without the pre-training.
  Neither has been run.
- Not decided: second seeds for P9, these two controls, and the other ablations.

## Round 2: runs around P9 (plan of 11/23 09:00 KST, written before the runs; decision of the user)

The other ablations of `fusion/attention/ablations.MD` change one thing in P0, which is below the baseline (0.242).
Instead of them, five runs that change one thing in P9 (`fusion_kp`), the run above the baseline. Protocol as in the
first round (700 demos, 101 epochs, last checkpoint, default settings, 660 episodes on each of `j_cells11` and
`k_cells11_heldout`, the sensor weights saved per step); the segmentation checkpoint is the `mix_ws` of the first
round, not retrained.

| run | relation to `ablations.MD` | what differs from P9 (`L0_vis_rgbdt_700demos_fusion_kp`, seed 42) |
|---|---|---|
| `L0_vis_rgbdt_700demos_fusion_kp_seed43`, `..._seed44` | P9, more seeds | the seed (43, 44) |
| `L0_vis_rgbdt_700demos_fusion_kp_scratch` | P1 on P9 | no segmentation pre-training: the same network, built as `mix_ws` was before its training (rgb encoder from ImageNet, depth and thermal encoders random), trained with the policy only. It uses no glass mask |
| `L0_vis_rgbdt_700demos_fusion_kp_frozen` | P2 on P9 | the pre-trained network is frozen; only the readout and the U-Net learn. The sensor weights stay those of `mix_ws` |
| `L0_vis_rgbd_700demos_fusion_kp` | P10 on P9 | no thermal image (the encoder treats thermal as missing) |

Overrides (after `training.seed=<seed> task.env_runner.test_set=j_cells11`), with
`PRE = policy.obs_encoder.pretrained=data/fusion_vis/runs/mix_ws/best.pt` and
`KP = policy.obs_encoder.readout=[avg,mask,centroid,area,max,gates,keypoint,ssm]`: seeds `PRE KP`; scratch
`++policy.obs_encoder.model_cfg={fusion_mode:weighted_sum,gate_norm:softmax} KP`; frozen
`PRE policy.obs_encoder.freeze_segmenter=true KP`; rgbd `PRE KP` with the variant `rgbd` of
`scripts/train_fusion_policy.sh`. Correction to the plan of the first round: of the encoders of the segmentation
network only the rgb one starts from ImageNet weights.

The runs are started one at a time as a GPU becomes free, behind the runs of
[`../cue_gate/README.md`](../cue_gate/README.md), in the order of the table. A run that fails is not started again.

**Read-out, fixed now.** "Same episodes" = the same test set and episode seed; sign test on the episodes in which
exactly one of the two succeeds; all cells and per cell.

1. *P9 with three seeds.* Macro average of each seed and their mean. The three seeds pooled against the three seeds
   of the rgbdt ImageNet baseline (seed 42 with 42, 43 with 43, 44 with 44) on the same episodes: P9 counts as above
   the baseline if the macro average is above with p < 0.05; the cells in which it is below with p < 0.05 are
   listed. Also listed, cell by cell, against the three seeds of the best baseline of that cell (the table of stage
   6): the cells in which P9 is below it with p < 0.05. P9 stays outside the comparison of stage 6 whatever the
   result (it uses the simulator's masks).
2. *Pre-training.* `fusion_kp_scratch` against P9 of seed 42, same episodes: if P9 is above with p < 0.05, the
   pre-training on masks contributes; otherwise it is not shown to matter. `fusion_kp_scratch` against the rgbdt
   ImageNet baseline of seed 42: if it is above with p < 0.05, the readout of P9 is above the baseline without any
   mask. Its share of episodes with a close within 1.5 cm of the glass is reported.
3. *Frozen encoder.* `fusion_kp_frozen` against P9 of seed 42, same episodes, in both directions with p < 0.05.
   Its sensor weights by cell and the AUC of each sensor's weight between the cells in which it shows the glass and
   those in which it does not are reported. If it is above P9, keeping the pre-trained network helps; this run
   cannot separate the kept weights from the kept features.
4. *Thermal.* P9 of seed 42 against `L0_vis_rgbd_700demos_fusion_kp`, same episodes, over all cells, over the cells
   in which thermal shows the glass and depth does not (2, 4, 9) and over those in which thermal does not show it
   (3, 5, 6, 7, 8): if P9 is above with p < 0.05 the thermal image contributes there; otherwise it is not shown to.

Points 2 to 4 have one seed each; no claim about the method is made from them alone. More seeds for them and the
other ablations are decided after this round.

### Results of round 2 (11/24 11:04 KST; 1320 test episodes per run, 120 per cell)

The five runs finished without a failed run (the last one 11/24 11:01 KST). All numbers are printed by
`python fusion/attention/readout_l0_vis_round2.py` ([`readout_round2.txt`](readout_round2.txt)); success per cell of
a run is in its `cells.json`.

| cell | sensors that show the glass | P9 seed 42 | seed 43 | seed 44 | no pre-training (`_scratch`) | frozen (`_frozen`) | no thermal (`rgbd`) | rgbdt ImageNet, seed 42 | best baseline (3 seeds, frozen) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | rgb+depth+thermal | 0.692 | 0.608 | 0.825 | 0.042 | 0.117 | 0.442 | 0.775 | 0.781 |
| 1 | depth+thermal | 0.625 | 0.517 | 0.608 | 0.033 | 0.167 | 0.400 | 0.750 | 0.758 |
| 2 | rgb+thermal | 0.550 | 0.442 | 0.567 | 0.033 | 0.167 | 0.067 | 0.225 | 0.397 |
| 3 | rgb+depth | 0.667 | 0.583 | 0.658 | 0.008 | 0.100 | 0.483 | 0.650 | 0.700 |
| 4 | thermal | 0.492 | 0.350 | 0.642 | 0.025 | 0.117 | 0.033 | 0.192 | 0.189 |
| 5 | depth | 0.583 | 0.533 | 0.650 | 0.058 | 0.158 | 0.508 | 0.650 | 0.697 |
| 6 | rgb | 0.492 | 0.342 | 0.533 | 0.008 | 0.158 | 0.042 | 0.158 | 0.256 |
| 7 | rgb (faint) | 0.367 | 0.350 | 0.417 | 0.008 | 0.100 | 0.083 | 0.092 | 0.181 |
| 8 | depth (faint) | 0.458 | 0.492 | 0.525 | 0.000 | 0.125 | 0.392 | 0.525 | 0.486 |
| 9 | thermal (faint) | 0.342 | 0.275 | 0.392 | 0.008 | 0.117 | 0.058 | 0.050 | 0.094 |
| 10 | all faint | 0.583 | 0.542 | 0.650 | 0.017 | 0.183 | 0.350 | 0.483 | 0.503 |
| | **macro average** | 0.532 | 0.458 | 0.588 | 0.022 | 0.137 | 0.260 | 0.414 | 0.458 |

The four runs other than the seeds are seed 42. Their training went through as that of P9 (training loss at the end
0.001 to 0.003; rollouts during training in line with the final numbers: no pre-training 0.00 to 0.05, frozen 0.02
to 0.18, no thermal 0.15 to 0.35), so the low numbers are those of the trained policies.

**1. P9 with three seeds.** Macro averages 0.532, 0.458 and 0.588, mean 0.526 (standard deviation 0.065). Pooled
against the same seeds of the rgbdt ImageNet baseline on the same episodes: 0.526 against 0.428, only P9 succeeds in
936 episodes, only the baseline in 550, p = 9.8e-24. P9 is above the baseline. It is below it with p < 0.05 in cells
0, 1 and 5. Against the three seeds of the best baseline of each cell: below in cells 0, 1, 3 and 5 (all cells in
which depth shows the glass fully), above in cells 2, 4, 6, 7, 9 and 10, no difference in cell 8. The seeds are
0.13 apart from the lowest to the highest.

**2. Pre-training.** Without the segmentation pre-training the policy does not work: 0.022 against 0.532 of P9 (only
P9 succeeds in 684 episodes, only this run in 11, p = 5.2e-186), below P9 in every cell. So the pre-training on
masks contributes; by the size it is what P9 rests on. Against the rgbdt ImageNet baseline of seed 42: 0.022 against
0.414 (12 / 529, p = 3.3e-139), below in ten of the eleven cells: the readout of P9 is not above the baseline
without the masks. The gripper closes within 1.5 cm of the glass in 0.015 of the episodes (P9 0.623, the baseline
0.364), at a median distance of 11.5 cm: without the masks this network does not find the glass at all.

**3. Frozen encoder.** With the pre-trained network frozen: 0.137 against 0.532 (44 / 565, p = 2.6e-116), below P9
in every cell, and below the baseline of seed 42 (0.137 against 0.414; 85 / 450, p = 5.5e-61) in the six cells in
which depth shows the glass. It closes within 1.5 cm of the glass in 0.129 of the episodes. Keeping the pre-trained
network as it is does not help; the features have to be trained with the policy. Its sensor weights are those of
`mix_ws`: the thermal weight is 0.17 to 0.27 in the cells in which thermal shows the glass and 0.01 to 0.02 in the
others (AUC 0.97); the rgb and the depth weight are equal in every cell (0.37 to 0.49 each) and do not follow what
the two sensors show (AUC 0.55 and 0.57). After the training with the policy (P9) the three weights are equal in
every cell (0.336 / 0.336 / 0.328). So the weighting that follows the thermal image exists only in the run that
does not work, and the run that works does not weight.

**4. Thermal.** P9 against P9 without the thermal image, seed 42: 0.532 against 0.260 over all cells (only P9
succeeds in 449 episodes, only the run without thermal in 90, p = 2.5e-58); 0.461 against 0.053 in the cells in
which thermal shows the glass and depth does not (2, 4, 9; 152 / 5, p = 8.4e-39); 0.513 against 0.302 in the cells
in which thermal does not show it (3, 5, 6, 7, 8; 172 / 45, p = 1.1e-18). By the rule fixed above the thermal image
contributes in all three. The third line needs a remark: in those cells the thermal image does not show the glass,
and the loss is in the two cells with rgb alone (6 and 7: 0.429 with thermal, 0.062 without) and in cell 3 (0.667
against 0.483); cells 5 and 8 do not differ. Next to the read-out, not fixed before the runs: the run without
thermal is also below the plain rgbd ImageNet baseline of seed 42 on the same episodes (0.260 against 0.404, only
the fusion run succeeds in 97, only the baseline in 287, p = 6.4e-23), with depth (0.429 against 0.621) and without
(0.057 against 0.143). So this run does not isolate what the thermal image adds: without it the method as a whole
works worse than a plain policy on the same inputs, also where thermal had nothing to show. One seed, and the seeds
of P9 are 0.13 apart.

**What round 2 says.**

- P9 is above the rgbdt baseline with three seeds (0.526 against 0.428), by the cells without depth; in four of the
  six cells in which depth shows the glass it is below the best baseline. It is outside the comparison of stage 6
  (simulator masks); inside it, it would not be "better in nearly every case".
- The gain needs both the pre-training on the simulator's masks and the training of the pre-trained network with
  the policy: without the first 0.02, without the second 0.14.
- Nothing in this round shows that weighting the sensors by what they show is what helps: the weights of P9 are
  equal in every cell, as in the first round.
- Points 2 to 4 are one seed each. Not decided: more seeds for them, and the other ablations of `ablations.MD`.
