# Hindsight cue gating on `L0_vis`

A gating method for the RGB-D-T Diffusion Policy on the visibility benchmark
([`../grid_bench/README.md`](../grid_bench/README.md), Stage 6). It decides per episode which sensors to trust and
where the glass is, and it learns both from the demonstrations alone: no cell, visibility flag or mask is used in
training or at test time.

Status (11/23 03:30 KST): the perception part is built and measured (below). The policy runs of round 1 are listed
here before they start; they start when the GPUs are free.

## Why this form

Three findings of the benchmark decide the design (`../grid_bench/README.md`, "Where the baselines fail"):

1. Success is decided by where the gripper closes: 0.97 within 0.5 cm of the glass, 0.46 at 1-1.5 cm, 0.01 beyond 4 cm.
2. Without depth the plain policies do not find the glass (in the cells without depth where rgb shows it fully, 2
   and 6, the gripper closes within 4 cm in 0.39 of the episodes for rgbdt and in 0.88 for the rgb-only reference).
3. A sensor that does not show the glass hurts: rgbdt is significantly below policies with fewer sensors in cells 2, 6
   and 7, and every cell's best baseline is below the single-sensor reference of that cell, which is told the cell.

So a method needs (a) to know which sensors show the glass in this episode, without being told, and (b) to hand the
policy the position those sensors agree on. A gate that only reweights features does (a); the cue does both.

## The method

```
first frame of the episode
    rgb ─► map net ─► where will the grasp be? p_rgb(u) ─► own peak ─► verifier: is there a glass at this peak? ─► r_rgb
  depth ─► map net ─► p_depth(u) ───────────────────────► own peak ─► verifier ─────────────────────────────────► r_depth
thermal ─► map net ─► p_thermal(u) ─────────────────────► own peak ─► verifier ─────────────────────────────────► r_thermal

cue  = [grasp point x, y, z ; pixel u, v ; confidence ; r_rgb, r_depth, r_thermal]      (kept for the whole episode)
       grasp point: the maps of the most trusted sensor and of the trusted sensors (r >= 0.5) that agree with it

every step
    rgb / depth / thermal ─► ResNet-18 each ─► feature of a sensor with r < 0.5 replaced by a learned embedding ─┐
    cue, joints ─────────────────────────────────────────────────────────────────────────────────────────────────┴─► Diffusion Policy
```

**Cue.** The demonstration itself says where the glass was: the tool position at the first step that closes the
gripper (arm kinematics), projected into the image with the camera calibration. Per sensor a small fully convolutional
network (local receptive field, no global pooling) learns a distribution over the 96 x 96 pixels of where the grasp
will be, and a map of the grasp height, from the first four frames of every demo: the arm has not moved yet, so the map
can only come from the scene, and a gripper near the table cannot leak the answer.

**Gate.** A sensor that does not show the glass still has to put its peak somewhere. Whether a peak is to be trusted is
learned without labels by cross-fitting: the map of every demo comes from a network that was not trained on it (4
folds), so a blind sensor is as wrong on it as it will be at test time, and "the peak was within 2 px of the hindsight
grasp pixel" is a label that costs nothing. Per sensor a small network, the verifier, sees the 25 x 25 image patch
around the sensor's own peak (and the map there) and predicts that label. Its output r_m is the gate. The verifier of a
fold never saw the fold's demos either, so the cue a policy is trained with has the errors it will have at test time.

**Gated policy.** Diffusion Policy with the encoders, denoiser and schedule of the plain baselines
(`policy/cue_gate_encoder.py` is DP's encoder class with two additions): the cue vector is part of the conditioning,
and the image feature of a sensor with r < 0.5 is replaced by a learned embedding (the most trusted sensor always stays
on). The rule is the same in training and at inference, so every set of sensors is a mode the policy was trained in.
In an evaluation the env computes the cue from the first frame of the episode (`policy/glass_gym_env.py`).

**Information used.** Sensor images, joint values and the demonstrated actions; arm kinematics, the camera calibration
and ImageNet weights for the policy's encoders. This is the information the comparison rules of Stage 6 allow. The
cells, flags and the position of the glass appear in the reports only.

**How it relates to what was tried.** The first cue gating of this repo ([`../../cue_gating/README.md`](../../cue_gating/README.md))
computed the gate from physical statistics of the whole frame; on this benchmark a frame looks the same whether or not
the glass is in it, so the cue has to be about the object. The learned attention weighting (`fusion/attention/`) learns
where the glass is from the simulator's masks; here the supervision is the demo's own grasp. Policy composition
(stage 4 of the baseline programme, 0.125) and random sensor dropout (stage 3, 0.246) did not fill the floor. The rule
that a fusion weight should follow a sensor's generalization error (Zhang et al., ICML 2023; Cao et al., ICML 2024) is
what cross-fitting measures directly, on the action the demonstrator took.

## Perception: results (11/23, no policy involved)

`python policy/hindsight_cue.py results/data/sim_demos_L0_vis_700.zarr --eval results/data/sim_demos_L0_vis_700_seed700.zarr --out data/hcue/L0_vis_700`
([`perception/verified_gate.txt`](perception/verified_gate.txt)). "Training demos" are the 700 demos with maps and
gates from networks that did not see them; "held-out" are 700 demos of other seeds, used for this report only.

| cell | shows the glass | training demos: fused peak within 1.5 px | within 1 cm on the table | within 1.5 cm | height within 2.5 cm | held-out: within 1.5 px | within 1 cm | within 1.5 cm | height within 2.5 cm |
|---|---|---|---|---|---|---|---|---|---|
| 0 | rgb+depth+thermal | 0.99 | 0.96 | 1.00 | 0.99 | 0.99 | 0.99 | 1.00 | 0.99 |
| 1 | depth+thermal | 1.00 | 0.89 | 0.97 | 1.00 | 1.00 | 0.93 | 0.99 | 1.00 |
| 2 | rgb+thermal | 1.00 | 0.94 | 0.97 | 1.00 | 1.00 | 0.94 | 0.99 | 0.99 |
| 3 | rgb+depth | 1.00 | 0.91 | 0.96 | 0.99 | 1.00 | 0.96 | 1.00 | 1.00 |
| 4 | thermal | 0.96 | 0.74 | 0.84 | 0.98 | 0.90 | 0.78 | 0.90 | 0.90 |
| 5 | depth | 1.00 | 0.86 | 0.96 | 0.98 | 0.98 | 0.80 | 0.92 | 0.98 |
| 6 | rgb | 1.00 | 0.84 | 0.90 | 0.98 | 1.00 | 0.98 | 1.00 | 0.98 |
| 7 | rgb faint | 1.00 | 0.82 | 0.90 | 1.00 | 1.00 | 0.96 | 0.98 | 1.00 |
| 8 | depth faint | 0.86 | 0.50 | 0.78 | 0.86 | 0.90 | 0.58 | 0.74 | 0.92 |
| 9 | thermal faint | 0.70 | 0.52 | 0.62 | 0.74 | 0.76 | 0.66 | 0.70 | 0.80 |
| 10 | all faint | 1.00 | 0.78 | 0.96 | 0.98 | 1.00 | 0.84 | 1.00 | 1.00 |
| all | | 0.96 | 0.83 | 0.92 | 0.96 | 0.97 | 0.88 | 0.94 | 0.97 |

The gate against what the cell says (the cell is not used to train it):

| sensor | own peak within 1.5 px where it shows the glass / faintly / not | gate r, median: shows / faintly / not | AUROC of r: shows against blind (training demos / held-out) |
|---|---|---|---|
| rgb | 0.99 / 1.00 / 0.10 | 1.00 / 1.00 / 0.18 | 0.999 / 0.998 |
| depth | 0.99 / 0.93 / 0.13 | 0.99 / 0.99 / 0.23 | 1.000 / 1.000 |
| thermal | 0.95 / 0.75 / 0.08 | 0.99 / 0.97 / 0.17 | 0.970 / 0.976 |

- Every sensor finds the grasp pixel where it shows the glass, trained on all demos (the blind ones do not hurt a
  local network). The open part was choosing among the sensors, and the verifier does it: the fused peak is within
  1.5 px in 0.96 of the demos, against 0.98 for always taking the best sensor in hindsight.
- With r >= 0.5 a sensor that shows the glass is on in 0.95-1.00 of the training demos (out of fold; thermal faint
  0.79) and a blind one in 0.03-0.15; 1.9 sensors are on per episode on average.
- The weak cells are the two where a single sensor shows the glass faintly and it is depth or thermal (8, 9).

**Forms compared.** Chosen on one number fixed beforehand: the share of training demos, out of fold, whose fused grasp
point is within 1 cm on the table. The held-out numbers were printed next to it and agree.

| form | within 1.5 px | within 1 cm | within 1.5 cm | record |
|---|---|---|---|---|
| gate from four numbers of every map and the peak distances; fusion mixture / product / pick | 0.900 / 0.904 / 0.904 | 0.730 / 0.731 / 0.716 | 0.830 / 0.830 / 0.840 | [`perception/gate_on_map_numbers.txt`](perception/gate_on_map_numbers.txt) |
| the three sensors trained together on the fused map, gate = a sensor's total evidence | 0.693 | 0.316 | 0.459 | [`perception/trained_together.txt`](perception/trained_together.txt) |
| verified gate, the most trusted sensor alone | 0.961 | 0.804 | 0.917 | [`perception/verified_gate.txt`](perception/verified_gate.txt) |
| verified gate, every trusted sensor | 0.923 | 0.780 | 0.867 | same |
| **verified gate, the most trusted sensor and the trusted ones that agree with it** | **0.964** | **0.827** | **0.916** | same |

- A gate that only sees numbers of the maps stops at 0.90: a blind sensor's peak can be as sharp as a real one.
- Trained together, rgb carries the fusion and depth and thermal stop learning (own peak within 1.5 px where the
  sensor shows the glass: 0.63 and 0.72, against 0.99 and 0.95 trained apart); the gate saturates at 1.
- Multiplying the maps of sensors that agree sharpens the point where several sensors show the glass (cells 0-3:
  within 1 cm 0.89-0.96 against 0.73-0.91 for the most trusted sensor alone); a trusted sensor that disagrees is left out.

**In the evaluation env** (`python policy/check_cue_env.py data/hcue/L0_vis_700/cue.pt --test_set <set>`,
[`perception/env_check.txt`](perception/env_check.txt)): the cue the env computes at reset against the position of
the glass, 660 episodes per test set. Run after the form was fixed; it checks the path the policies are evaluated
through and was not used to choose anything (an earlier form was checked the same way on `j_cells11` before the
verifier existed: within 1.5 cm 0.80).

| test set | within 1 cm | within 1.5 cm | within 4 cm | weakest cells (within 1.5 cm) |
|---|---|---|---|---|
| `j_cells11` | 0.85 | 0.93 | 0.99 | cell 9: 0.70, cell 8: 0.85 |
| `k_cells11_heldout` | 0.78 | 0.88 | 0.94 | cell 9: 0.67, cell 8: 0.75 |

Limits of this part: one seed of the networks; rerunning the verifier changes the gate of a few demos (GPU
non-determinism), so the files the policies use are fixed: `data/hcue/L0_vis_700/cue.pt` (sha256 `f2938d96...`) and
`train_cue.npy` (`9aba05f8...`), not in git. The cue is computed once, from the first frame; frames later in the
episode are not used.

## Policy runs, round 1 (listed 11/23 03:30 KST, before any of them starts)

Protocol of the frozen baselines: `L0_vis`, 700 demos, 101 epochs, last checkpoint, ImageNet encoders, rollouts during
training on `j_cells11` (55 episodes), final evaluation on `j_cells11` and `k_cells11_heldout` (660 episodes each, 120
per cell), seed 42. Queue `data/runs/cg_queue.sh`, at most three runs at once, results in
`results/hard_scenario/cue_gate/<run>/`.

| run | arm (`policy/train_cuegate.sh`) | cue in the conditioning | sensors' features gated | question |
|---|---|---|---|---|
| `L0_vis_rgbdt_700demos_hcg_full` | `full` | yes | yes | the method |
| `L0_vis_rgbdt_700demos_hcg_cue` | `cue` | yes | no | does gating the features add to the cue? |
| `L0_vis_rgbdt_700demos_hcg_gate` | `gate` | no (only which sensors are on) | yes | does the gate alone help, without the grasp point? |
| `L0_vis_rgbdt_700demos_hcg_blind` | `blind` | yes | no images at all | do the images add anything to the cue and the joints? |

The plain rgbdt baseline (no cue, no gate) is the fourth corner and exists with three seeds (0.428).

**Read-out, fixed now.**

1. Against the baselines, per run: success over the 1320 episodes against `L0_vis_rgbdt_700demos_imagenet` (seed 42,
   0.414) on the same episodes (sign test), and per cell against the seed-42 run of that cell's best baseline
   (Stage 6 table). "Above" at p < 0.05. One seed is a first reading: the claim of Stage 6 ("better in nearly every
   case": no cell significantly below its best baseline, macro above every baseline) is judged with three seeds.
2. Which part carries it: `full` against `cue`, `full` against `gate`, `cue` against `blind`, on the same episodes,
   p < 0.05.
3. The cue inside the episodes (`python policy/analyze_cue.py <run>`): success with a cue within 1.5 cm of the glass and
   with a cue further off, per run. This separates failures of the cue from failures of the policy.
4. Reported next to them, not criteria: the single-sensor references that are told the cell (0.674), and the runs of
   the learned attention weighting on the same episodes (`../fusion_vis/`, extra supervision).
5. Seeds 43 and 44 are then run for the arm with the highest success of round 1 (`full` if the best arms are within
   p >= 0.05 of each other). Further ablations (gate rule `top1` / `soft`, the gate on map numbers, the cue without
   cross-fitting, a cue from the simulator as an upper bound) are specified after round 1, before they start.

Results are reported as they come out, whatever they are.

## Files

- `policy/hindsight_cue.py` (cue, verifier, fusion; fit and report), `policy/check_cue_env.py` (the cue in the env),
  `policy/cue_gate_encoder.py` (gated encoder), `policy/config/train_glass_cuegate_unet_image.yaml`,
  `policy/config/task/glass_pick_hcue.yaml`, `policy/train_cuegate.sh` (arms), `policy/analyze_cue.py` (read-out 3).
- `policy/glass_dataset.py`, `policy/glass_gym_env.py`: the `hcue` observation (the cue of the episode); nothing changes
  for a policy without that key.
- `perception/`: the reports of this page. Not in git: `data/hcue/` (networks, maps, cue files).
