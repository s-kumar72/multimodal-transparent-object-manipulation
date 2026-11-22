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
