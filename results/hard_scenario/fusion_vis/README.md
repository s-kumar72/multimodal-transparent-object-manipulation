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
