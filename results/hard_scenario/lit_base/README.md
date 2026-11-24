# Baselines from the literature on the visibility benchmark `L0_vis`

Started 11/25 (approved by the user 11/25, step 5 of five). Until now every comparison of the benchmark was against
plain Diffusion Policy with the sensors' features concatenated (static fusion), and against methods of this project.
This folder adds fusion methods from the literature on multi-sensor imitation learning, under the protocol of the
frozen baselines (Stage 6 of [`../grid_bench/README.md`](../grid_bench/README.md)), so that they stand in the same
table.

## Which, and why these

The works on multi-sensor manipulation policies compare a small set of fusions, and each new one against the earlier:
concatenation of the per-sensor features (static; this is the plain baseline here), fusion by self-attention over the
sensors' feature tokens (MulSA: Li et al., "See, Hear, and Feel", CoRL 2022), and stage-guided dynamic fusion (MS-Bot:
Feng et al., "Play to the Score", CoRL 2024, which compares with concatenation, an LSTM fusion and MulSA). Random
dropout of whole sensors in training is the usual remedy for a policy that leans on one sensor and is part of several
recent fusion modules.

| baseline | what it is here | status |
|---|---|---|
| self-attention fusion (MulSA) | `policy/attn_fusion_encoder.py`, `policy/train_mulsa.sh`: the tokens are the ResNet features of every sensor at every observation step (3 x 2), one multi-head self-attention across sensors and time, the output tokens concatenated; they condition the denoiser of Diffusion Policy as the concatenated features of the plain baseline do | implemented, listed below |
| sensor dropout | `policy/train_dropout.sh` (stage 3 of the baseline programme, there without ImageNet encoders: 0.246 over three seeds): each sensor's image is blank with probability 0.3 in a training sample, at least one stays | listed below, now with ImageNet encoders |
| stage-guided dynamic fusion (MS-Bot) | `policy/msbot_encoder.py`, `policy/train_msbot.sh`: a state token from the sensors' feature tokens and the joints, learnable stage tokens (reach, grasp, lift; set to the mean state token of their stage after one epoch), a gate that scores the stages with the paper's soft constraint, the stage-injected state token as the query of a cross-attention over the feature tokens, random attention blur; the fused token and the joints condition the denoiser | implemented, listed below (11/25 01:48 KST, before it starts) |

What stays the same in all of them: Diffusion Policy's denoiser, schedule and optimizer, one ResNet-18 per sensor
started from ImageNet, crops, 700 demos of `L0_vis`, 101 epochs, last checkpoint. The papers' own action heads (an
MLP on discrete actions) are not used: the fusion is the thing compared, and the action head is the one of every
other policy of the benchmark. What the MulSA paper leaves open (number of layers and heads, normalization,
embeddings) is set in `policy/attn_fusion_encoder.py` and said there: one layer, 8 heads, LayerNorm before it, a
residual connection, no positional embedding.

MS-Bot, what is the paper's and what is not (`policy/msbot_encoder.py`): the paper's are the four parts, the gate
loss with weight 5, beta 0.5 and the attention blur with probability 0.25. Different here: the state token reads the
joint values of the two observation steps where the paper runs an LSTM over the action history (the policies of the
benchmark observe two steps and no past actions); the stages are a rule on the demonstrated actions (reach: the
gripper is commanded open, 63% of the frames; grasp: closed and the arm's target still, 13%; lift: after that, 24%),
as the paper's rule-based labels are, and a frame within one step of a stage change is not constrained; 8 heads and
the MLP sizes are not given in the paper. The stage labels use the demonstrated actions only, which the comparison
rules allow.

## Runs (listed 11/25 01:38 KST, before any of them starts)

Seed 42; `GROUP=lit_base TOPK=0 NTEST=55 NEVAL=660 TEST_SETS="j_cells11 k_cells11_heldout" TAG=<tag> TRAIN=<script>
EXTRA="training.seed=42 task.env_runner.test_set=j_cells11 policy.obs_encoder.rgb_model.weights=IMAGENET1K_V1" bash
scripts/run_experiment.sh L0_vis rgbdt 700`, then `policy/eval_new_cells.sh lit_base/<run>` (the 15 visibility
combinations that are not in the training set, 600 episodes on each of the two test sets).

| run | `TRAIN` |
|---|---|
| `L0_vis_rgbdt_700demos_mulsa` | `policy/train_mulsa.sh` |
| `L0_vis_rgbdt_700demos_dropout_imagenet` | `policy/train_dropout.sh` |
| `L0_vis_rgbdt_700demos_msbot` (listed 11/25 01:48 KST) | `policy/train_msbot.sh` |

Seeds 43 and 44 of each follow after the first reading, so that they stand in the table of Stage 6 with three seeds
like the plain baselines. A run that fails or is stopped is reported as such and not started again.

## Read-out, fixed now (`python policy/readout_lit_base.py`; per run, seed 42, same episodes, sign test, p < 0.05)

1. Against the plain rgbdt baseline (`L0_vis_rgbdt_700demos_imagenet`, 0.414): over the 1320 episodes and per cell;
   and per cell against that cell's best plain baseline.
2. Against the method: `hcg_full` (0.789) and `hcg_cue` (0.795) of seed 42 over the 1320 episodes and per cell.
3. On the 15 new combinations (1200 episodes): against the rgbdt baseline (0.574) and against `hcg_cue` (0.849).
4. Where the gripper first closes: the share of episodes with the first close within 1.5 cm of the glass, next to
   the plain baseline and the method (does the fusion find the glass more often?).

The baselines enter the Stage 6 comparison as baselines: the claim of the method ("no cell significantly below its
best baseline, macro above every baseline") is then judged against them too, with three seeds.
