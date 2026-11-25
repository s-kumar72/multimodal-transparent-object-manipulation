# A second policy on the visibility benchmark: the cue with another action head

Started 11/26 (decision of the user 11/26). Every policy of the benchmark so far is Diffusion Policy with the
UNet denoiser: the baselines, the method and its controls differ in what conditions that denoiser. This folder asks
whether the hindsight cue of [`../cue_gate/README.md`](../cue_gate/README.md) helps a policy of another kind in the
same way, so that the method is not a property of one action head.

## The policy

`policy/act_policy.py`: the action head of ACT (Zhao et al., "Learning Fine-Grained Bimanual Manipulation with
Low-Cost Hardware", RSS 2023) in place of the denoiser. No diffusion: a conditional VAE over the action chunk and a
transformer encoder-decoder with one learned query per action, trained with an L1 loss and the KL term (512 wide, 8
heads, 4 encoder layers, 1 decoder layer, feed-forward 3200, dropout 0.1, latent 32, KL weight 10: ACT's sizes).

What is not ACT's, because the benchmark fixes it for every policy: the observation tokens are the outputs of the
benchmark's encoder, one per observation step (pooled ResNet-18 features of each sensor started from ImageNet, the
joints, and the cue when the encoder has it), not feature-map tokens per camera; the chunk is the benchmark's horizon
(16 steps, 8 executed, then a new chunk) instead of 100 steps with temporal ensembling; optimizer, schedule, batch
size, EMA and the 101 epochs are the benchmark's. None of the head's settings was tuned on the benchmark.

The cue reaches this head as it reaches the denoiser in the arm `cue`: in the conditioning vector, image features not
gated (`policy/config/train_glass_act_cue_image.yaml`; the cue networks are those of the method, `data/hcue/L0_vis_700`).

**Checks before the listing (11/26).** One batch, 300 steps: the loss falls for both arms and the chunk predicted with
z = 0 approaches the batch (L1 0.15 and 0.19 in normalized units from 0.65). Two debug epochs through the training
workspace with rollouts, a checkpoint and the final evaluation of four episodes: both arms run through. No training
run of this folder has started.

## Runs (listed 11/26 01:02 KST, before any of them starts)

| runs | arm |
|---|---|
| `L0_vis_rgbdt_700demos_act`, `..._act_seed43`, `..._act_seed44` | `plain`: colour, depth, thermal, joints |
| `L0_vis_rgbdt_700demos_act_cue`, `..._act_cue_seed43`, `..._act_cue_seed44` | `cue`: the same with the hindsight cue in the conditioning |

Command: `GROUP=second_policy TOPK=0 NTEST=55 NEVAL=660 TEST_SETS="j_cells11 k_cells11_heldout" TAG=<act |
act_cue>[_seed<seed>] TRAIN="policy/train_act.sh <plain | cue>" EXTRA="training.seed=<seed>
task.env_runner.test_set=j_cells11" bash scripts/run_experiment.sh L0_vis rgbdt 700 && bash policy/eval_contacts.sh
<run>` (the 700 demos, 101 epochs, last checkpoint, 660 episodes on each of `j_cells11` and `k_cells11_heldout`, then
the same checkpoint on `L0_vis_ell`). Through the queue of the fusion-vis worktree, after everything queued before
them. A run that fails or is stopped is reported as such and not started again.

## Read-out, fixed now (`python policy/readout_second_policy.py > readout.txt`; tested with runs of Diffusion Policy standing in)

On `L0_vis` and on `L0_vis_ell`; runs paired by seed on the same episodes, two-sided sign test, p < 0.05.

1. The macro average of every run per seed, next to Diffusion Policy without the cue (the rgbdt baseline) and with it
   (`hcg_cue`).
2. *The cue in the second policy.* `act_cue` against `act` per seed and pooled, and per cell. "The cue carries over to
   the second policy" **holds over the seeds** when the pooled test is significant and `act_cue` is above `act` in
   each of the three seeds, **does not hold** when the pooled test is not significant, and is **mixed** otherwise.
   The verdict for the paper is the one on `L0_vis_ell`.
3. Where the gripper first closes (within 1.5 cm of the glass), as for the baselines from the literature.
4. Next to it, not a criterion: the two heads against each other, without the cue and with it.

If the plain run of this head stays near zero (below 0.10 on `L0_vis_ell` in every seed), that is said first: the
head would then not be trained on this benchmark with the benchmark's schedule, and a gain from the cue over it would
be a gain over a policy that does not work.

**Not in this folder.** The one-network control, the other controls and the other input sets with this head; this
head on the place level; a third policy.

## Files

`policy/act_policy.py`, `policy/config/train_glass_act_image.yaml`, `policy/config/train_glass_act_cue_image.yaml`,
`policy/train_act.sh`, `policy/readout_second_policy.py`.
