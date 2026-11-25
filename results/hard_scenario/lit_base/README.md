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

## Results of the first reading (11/25 19:35 KST; seed 42, 1320 episodes per run on the training cells, 1200 on the new combinations)

The three runs trained and were evaluated as listed (MulSA 06:18 to 11:30 KST, MS-Bot 13:16 to 18:47, sensor dropout
13:48 to 19:28); none failed. The read-out is [`readout.txt`](readout.txt) (`python policy/readout_lit_base.py`). The
same runs on `L0_vis_ell`, the pick level with the contact model of the place level (item 3 of round 3 of
[`../contact_check/README.md`](../contact_check/README.md)), are in [`readout_ell.txt`](readout_ell.txt)
(`PYTHONPATH=$PWD/policy python policy/readout_contacts_round3.py`, part B; its part A is the one reported there).

| run | `L0_vis` | against plain rgbdt (0.414): only the run / only plain, p | new combinations | against plain rgbdt (0.574), p | `L0_vis_ell` | against plain rgbdt (0.560), p | first close within 1.5 cm of the glass (`L0_vis`) |
|---|---|---|---|---|---|---|---|
| self-attention fusion (MulSA) | 0.412 | 135 / 137, 0.95 | 0.579 | 0.77 | 0.564 | 0.74 | 0.473 |
| sensor dropout | 0.410 | 167 / 172, 0.83 | 0.553 | 0.2 | 0.556 | 0.83 | 0.504 |
| stage-guided fusion (MS-Bot) | 0.161 | 67 / 401, 4.9e-59 | 0.220 | 1.4e-86 | 0.273 | 2.2e-63 | 0.167 |
| plain rgbdt | 0.414 | | 0.574 | | 0.560 | | 0.465 |
| `hcg_cue` | 0.795 | | 0.849 | | 0.914 | | 0.875 |

**1. Against the plain baselines.**

- *Self-attention fusion* is level with plain rgbdt on both levels and differs from it in no cell. Against the best
  plain baseline of the cell it is below in one cell (2 on `L0_vis`, 6 on `L0_vis_ell`) and above in none.
- *Sensor dropout* is level with plain rgbdt over all episodes on both levels. It moves success between cells: above
  plain rgbdt in cells 2 and 7 (the colour camera shows the glass and depth does not) on both levels, below it in
  cell 1 on `L0_vis` and in cells 0 and 5 on `L0_vis_ell` (depth shows the glass). Against the best plain baseline
  of the cell it is below in cell 1 (`L0_vis`) and in cells 0, 5 and 6 (`L0_vis_ell`), above in none.
- *MS-Bot* is below plain rgbdt in eight cells on each level and below the best plain baseline of every cell.

**2. Against the method.** All three are below `hcg_full` and `hcg_cue` on both levels (p < 1e-90) and above them in
no cell. On `L0_vis` self-attention fusion and sensor dropout are below `hcg_cue` in all eleven cells and below
`hcg_full` in ten; on `L0_vis_ell` self-attention fusion is below in eight (`hcg_full`) and nine (`hcg_cue`), sensor
dropout and MS-Bot in all eleven.

**3. The 15 new combinations.** Self-attention fusion 0.579 and sensor dropout 0.553 against plain rgbdt 0.574 (not
separated); MS-Bot 0.220. All are below `hcg_cue` (0.849; p < 1e-50).

**4. The first close of the gripper.** Neither fusion finds the glass more often than plain concatenation: the first
close is within 1.5 cm of the glass in 0.47 (self-attention) and 0.50 (dropout) of the episodes against 0.47; the
method is at 0.87. After such a close sensor dropout succeeds less often than plain rgbdt (0.69 against 0.75).
MS-Bot's first close is a median of 4.1 cm from the glass (the others 1.5 to 1.7 cm).

**The MS-Bot run, next to the read-out.** The run is the listed one and did not fail, but its numbers are not those
of a policy trained to the end:

- Its training loss (policy loss + 5 x gate loss) stays at 1.185 from epoch 5 to epoch 37, falls to 0.05 between
  epochs 38 and 44 and ends at 0.037. In the demos the stages are 0.630 (reach), 0.129 (grasp) and 0.241 (lift) of
  the frames, 0.614, 0.097 and 0.225 without the frames next to a stage change. A gate that gives the lift stage no
  score has a gate loss of 0.225, which is 1.13 of the 1.185: the plateau is what a gate that does not tell lift from
  the other stages costs. This is read from the loss; no checkpoint of that phase exists (only the last one is kept).
- The action error on training batches falls throughout, the plateau included, and ends at 0.0009 (sensor dropout
  0.0008, self-attention fusion 0.0004).
- The success in the 55 test episodes during training is 0.04, 0.07, 0.11 and 0.15 at epochs 25, 50, 75 and 100:
  still rising at the end. That of self-attention fusion is 0.33, 0.45, 0.51 and 0.44.

So with the protocol of the benchmark (101 epochs, last checkpoint) this implementation of MS-Bot spends the first
third of the training with a gate that does not work and is not trained out at the end. Whether a longer training or
a change of the gate loss would bring it to the level of the others was not tried: that would be another run than
the listed one, and is for the user to decide.

**What this says.**

- Fusing the sensors' features by self-attention, or dropping sensors in training, leaves the policy where plain
  concatenation is: on the training cells, on the new combinations and with the contact model of the place level.
  Neither closes the gripper nearer to the glass. What the method changes (the grasp point in the conditioning) is
  not something these two fusions supply.
- The number of MS-Bot is far below and should not stand in a table as "MS-Bot" without the remark above.
- The method is above all three in almost every cell and in no cell below.

**What it does not say.** One seed each. The settings of the baselines are the papers' or the stated defaults; none
was tuned on the benchmark.

## Seeds 43 and 44 (listed 11/25 19:35 KST, before any of them starts)

As announced under "Runs": seeds 43 and 44 of the baselines, so that they stand in the table of Stage 6 with three
seeds. Four runs, each as its run of seed 42 with `training.seed=<seed>` and `TAG=<tag>_seed<seed>`, followed by
`bash policy/eval_new_cells.sh lit_base/<run>` and `bash policy/eval_contacts.sh <run>` (`L0_vis_ell`; the results
are in `../contact_check/<run>/`):

| run | `TRAIN` |
|---|---|
| `L0_vis_rgbdt_700demos_mulsa_seed43`, `L0_vis_rgbdt_700demos_mulsa_seed44` | `policy/train_mulsa.sh` |
| `L0_vis_rgbdt_700demos_dropout_imagenet_seed43`, `L0_vis_rgbdt_700demos_dropout_imagenet_seed44` | `policy/train_dropout.sh` |

They are trained in the grid-bench worktree through the queue, after everything queued before them. A run that fails
is reported and not started again.

**MS-Bot: its seeds are held.** Two more seeds of the run above would cost about eleven GPU hours and say how much
the plateau varies, not how the method does. They are not started until the user decides between leaving MS-Bot out
of the table, keeping the listed run with the remark, or listing a changed run.

**Read-out with three seeds, fixed now** (`PYTHONPATH=$PWD/policy python policy/readout_lit_seeds.py >
readout_seeds.txt`; tested on the runs of seed 42 with `--seeds 42`). On `L0_vis` and on `L0_vis_ell`, runs paired by
seed on the same episodes and pooled, sign test, p < 0.05:

1. The macro average of every run per seed, mean and standard deviation.
2. Each baseline from the literature against the plain rgbdt baseline, over all episodes and per cell.
3. Rules 4 and 5 of Stage 6 with these baselines among the baselines. The best baseline of a cell is the one with the
   highest success over its three seeds among the four plain baselines and the baselines from the literature that
   have three seeds (a tie goes to the plain one). "Better in nearly every case" **holds** for a method run on a
   level when no cell is below its best baseline and the macro average is above that of every baseline, as in Stage
   6. It is read for `hcg_full` and, once round 4 of [`../cue_gate/README.md`](../cue_gate/README.md) has given it
   three seeds, for `hcg_cue`; the verdict for the paper's table is the one on `L0_vis_ell`.
4. The 15 new combinations (`L0_vis`): each baseline from the literature against `hcg_cue`, the seeds that both have.

**Known when this is listed.** The results of seed 42 above. From the test of the script on seed 42: with
self-attention fusion and sensor dropout among the baselines, the best baseline of cells 3 and 10 on `L0_vis` and of
cell 2 on `L0_vis_ell` is one of the two, and no cell of `hcg_full` or `hcg_cue` is below its best baseline on
either level.
