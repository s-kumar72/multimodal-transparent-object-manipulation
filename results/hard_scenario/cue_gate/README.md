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

## Results of round 1 (11/23 13:35 KST; seed 42, 1320 test episodes per run, 120 per cell)

`python policy/readout_cue_gate.py` ([`readout.txt`](readout.txt)) and `python policy/analyze_cue.py <run>`
([`cue_in_episodes.txt`](cue_in_episodes.txt)); all four runs finished and were evaluated. "(above)": above the seed-42
run of the best baseline of the cell on the same episodes with p < 0.05; in the last row against the rgbdt ImageNet
baseline of seed 42 over all episodes. No arm is below the best baseline of a cell with p < 0.05.

| cell | sensors that show the glass | `full` | `cue` | `gate` | `blind` | best baseline of the cell, its seed-42 run | the same, 3 seeds (frozen) | reference |
|---|---|---|---|---|---|---|---|---|
| 0 | rgb+depth+thermal | 0.942 (above) | 0.950 (above) | 0.767 | 0.942 (above) | 0.775 (rgbdt) | 0.781 | 0.900 |
| 1 | depth+thermal | 0.883 (above) | 0.858 (above) | 0.725 | 0.867 (above) | 0.750 (rgbdt) | 0.758 | 0.875 |
| 2 | rgb+thermal | 0.883 (above) | 0.900 (above) | 0.508 | 0.900 (above) | 0.408 (rgbt) | 0.397 | 0.642 |
| 3 | rgb+depth | 0.925 (above) | 0.933 (above) | 0.733 | 0.917 (above) | 0.700 (rgbd) | 0.700 | 0.867 |
| 4 | thermal | 0.617 (above) | 0.642 (above) | 0.283 (above) | 0.617 (above) | 0.167 (rgbt) | 0.189 | 0.608 |
| 5 | depth | 0.733 | 0.792 (above) | 0.683 | 0.692 | 0.633 (rgbd) | 0.697 | 0.858 |
| 6 | rgb | 0.900 (above) | 0.883 (above) | 0.325 | 0.900 (above) | 0.275 (rgb) | 0.256 | 0.492 |
| 7 | rgb (faint) | 0.892 (above) | 0.842 (above) | 0.217 | 0.875 (above) | 0.167 (rgb) | 0.181 | 0.333 |
| 8 | depth (faint) | 0.558 | 0.625 | 0.617 | 0.533 | 0.525 (rgbdt) | 0.486 | 0.742 |
| 9 | thermal (faint) | 0.525 (above) | 0.467 (above) | 0.175 | 0.508 (above) | 0.108 (rgbt) | 0.094 | 0.383 |
| 10 | all faint | 0.817 (above) | 0.858 (above) | 0.542 | 0.858 (above) | 0.483 (rgbdt) | 0.503 | 0.717 |
| macro average | | 0.789 (above) | 0.795 (above) | 0.507 (above) | 0.783 (above) | 0.454 | 0.458 | 0.674 |

**1. Against the baselines.** Against the rgbdt ImageNet baseline of seed 42 (0.414) over all episodes: `full` 0.789
(only the arm succeeds in 567 episodes, only the baseline in 72, p = 2.5e-96), `cue` 0.795 (568 / 64, p = 6.5e-102),
`blind` 0.783 (564 / 77, p = 2e-92), `gate` 0.507 (258 / 135, p = 5.5e-10). Per cell against the seed-42 run of the
best baseline: `full` is above in 9 cells (all but 5 and 8), `cue` in 10 (all but 8), `blind` in 9 (all but 5 and 8),
`gate` in one (4); none is below anywhere. One seed: a first reading; the claim of Stage 6 waits for three seeds.

**2. Which part carries it.** `full` against `cue`: 0.789 against 0.795 (98 / 107, p = 0.58), no cell differs: gating
the image features adds nothing to the cue. `full` against `gate`: 0.789 against 0.507 (458 / 86, p = 2.7e-62): the
grasp point in the conditioning is what carries the method. `cue` against `blind`: 0.795 against 0.783 (112 / 95,
p = 0.27); one cell differs (5, depth only: 0.792 against 0.692, p = 0.017): with the cue and the joints, the images
are not shown to add anything over all cells. The gate without the grasp point is above the baseline (0.507 against
0.414), mostly where depth does not show the glass (0.302 against 0.143 there).

**3. The cue inside the episodes.** The cue is within 1.5 cm of the glass in 0.91 of the episodes (1197 of 1320; the
same for every arm, it is computed from the first frame). Success with a right cue / with a wrong cue: `full` 0.843 /
0.260, `cue` 0.855 / 0.220, `blind` 0.835 / 0.268, `gate` 0.535 / 0.236. The cells with the fewest right cues are 9
(0.68) and 8 (0.80). With a right cue the success is lowest in the cells in which only depth or only thermal shows
the glass (4, 5, 8, 9: 0.62 to 0.83 for `cue`).

**4. Next to them, not criteria.** References (told the cell, one seed): 0.674 on average; the arms with a cue are
above it on average and below it in cells 5 and 8. Learned attention weighting on the same episodes (extra
supervision): P9 0.532, P0 0.242, P5 0.155; `full` against P9: 437 / 98, p = 4.4e-52.

**5. Seeds.** `cue` has the highest success; `full`, `cue` and `blind` are within p >= 0.05 of each other (`full`
against `blind` p = 0.60), so by the rule the seeds 43 and 44 are run for `full`
(`L0_vis_rgbdt_700demos_hcg_full_seed43`, `..._seed44`, queued 11/23 13:36 KST).

Outside the read-out, where the gripper closes (last close): within 1.5 cm of the glass in 0.86 of the episodes for
`full`, 0.87 `cue`, 0.86 `blind`, 0.57 `gate`, 0.36 for the rgbdt baseline; in the cells without depth 0.85 / 0.84 /
0.84 / 0.33 / 0.08.

**What round 1 says** (one seed):

- A grasp point computed once from the first frame, learned from the demos' own hindsight with no simulator label,
  takes the plain policy from 0.41 to 0.79 on `L0_vis`, with no cell below the best baseline.
- Once the policy has that point it does not need the images (`blind`), and gating their features changes nothing
  (`full` against `cue`): on this benchmark the outcome is decided by finding the glass before the arm moves.
- So the gate of the method can only matter inside the cue, in deciding which sensors' maps make the grasp point.
  Rounds 2 and 3 measure that.

## Results of the seeds: `full` with seeds 42, 43 and 44 against the rules of Stage 6 (11/23 21:35 KST)

Item 5 of round 1. Full tables: [`readout_seeds.txt`](readout_seeds.txt) (`python policy/readout_cue_seeds.py`). The
method's run of a seed is paired with the baseline's run of the same seed on the same episodes; the three pairs are
pooled (360 episodes per cell, 3960 in all), sign test on the episodes only one of the two solves.

| cell | sensors that show the glass | seed 42 | seed 43 | seed 44 | the three seeds | best baseline of the cell (frozen) | only the method succeeds / only the baseline | p | reference |
|---|---|---|---|---|---|---|---|---|---|
| 0 | rgb+depth+thermal | 0.942 | 0.925 | 0.950 | 0.939 (above) | 0.781 (rgbdt) | 66 / 9 | 7.7e-12 | 0.900 |
| 1 | depth+thermal | 0.883 | 0.917 | 0.925 | 0.908 (above) | 0.758 (rgbdt) | 73 / 19 | 1.2e-08 | 0.875 |
| 2 | rgb+thermal | 0.883 | 0.900 | 0.917 | 0.900 (above) | 0.397 (rgbt) | 188 / 7 | 7.9e-47 | 0.642 |
| 3 | rgb+depth | 0.925 | 0.925 | 0.908 | 0.919 (above) | 0.700 (rgbd) | 92 / 13 | 8e-16 | 0.867 |
| 4 | thermal | 0.617 | 0.675 | 0.692 | 0.661 (above) | 0.189 (rgbt) | 184 / 14 | 5.5e-39 | 0.608 |
| 5 | depth | 0.733 | 0.750 | 0.725 | 0.736 | 0.697 (rgbd) | 67 / 53 | 0.24 | 0.858 |
| 6 | rgb | 0.900 | 0.908 | 0.892 | 0.900 (above) | 0.256 (rgb) | 236 / 4 | 1.6e-64 | 0.492 |
| 7 | rgb (faint) | 0.892 | 0.833 | 0.867 | 0.864 (above) | 0.181 (rgb) | 250 / 4 | 1.2e-68 | 0.333 |
| 8 | depth (faint) | 0.558 | 0.517 | 0.567 | 0.547 | 0.486 (rgbdt) | 81 / 59 | 0.076 | 0.742 |
| 9 | thermal (faint) | 0.525 | 0.533 | 0.483 | 0.514 (above) | 0.094 (rgbt) | 156 / 5 | 6e-40 | 0.383 |
| 10 | all faint | 0.817 | 0.833 | 0.833 | 0.828 (above) | 0.503 (rgbdt) | 137 / 20 | 1.2e-22 | 0.717 |
| macro average | | 0.789 | 0.792 | 0.796 | 0.792 | 0.458 | | | 0.674 |

- **Rule 5, first condition** (no cell below its best baseline with p < 0.05): met. Above in nine cells; in the two
  cells where depth alone shows the glass (5, and 8 where it is faint) the method is 0.04 and 0.06 above and that is
  not significant (p = 0.24 and 0.076).
- **Rule 5, second condition** (macro average above every baseline): met. 0.792 against rgbdt 0.428 (only the method
  succeeds 1656, only the baseline 214, p = 5.9e-276), rgbd 0.397, rgbt 0.194, rgb 0.153.
- So "better in nearly every case" holds for `full` under the rules as frozen on 11/22. The three seeds are 0.789,
  0.792 and 0.796 (standard deviation 0.004; the three seeds of the rgbdt baseline: 0.414, 0.440, 0.431).
- Not criteria: the method is above the reference (single-sensor policies that are told the cell, one seed) in nine
  cells and below it in 5 (0.736 against 0.858) and 8 (0.547 against 0.742), the same two cells. Against the learned
  attention weighting P9 with the same three seeds (extra supervision, [`../fusion_vis/README.md`](../fusion_vis/README.md)):
  0.792 against 0.526 (1348 / 292, p = 5.7e-162), above in ten cells, cell 8 not significant (0.547 against 0.492,
  p = 0.11).

What this does not show: which part of the method the result needs (rounds 2 and 3; `cue` and `blind` of round 1 are
as high with one seed), and anything outside the eleven cells of the training set (round 3).

## Policy runs, round 2: what the cue is made of, and two controls (listed 11/23 12:20 KST, before any of them starts)

Written with three of the four runs of round 1 finished and looked at (`blind` still running): `full` 0.789, `cue`
0.795, `gate` 0.507, the rgbdt baseline of the same seed 0.414; `full` against `cue` p = 0.58. Gating the image
features adds nothing once the cue is in the conditioning, so the part of the gate that can matter is the one inside
the cue: the verifier decides which sensors' maps make the grasp point. That has been measured on the cue alone
(table "Forms compared"), not on the success of a policy. Round 2 measures it there, and adds two controls that use the
same hindsight label without the per-sensor maps and gate. Decision of the user (11/23).

Protocol of round 1 (`L0_vis`, 700 demos, 101 epochs, last checkpoint, ImageNet encoders, 55 episodes of `j_cells11`
during training, final evaluation on `j_cells11` and `k_cells11_heldout`, 660 episodes each), seed 42. The first
three runs are the arm `cue` of `policy/train_cuegate.sh` (cue in the conditioning, image features not gated) with
another cue folder (`task.hcue_dir`), so against `L0_vis_rgbdt_700demos_hcg_cue` only the cue differs.

| run | cue | question |
|---|---|---|
| `L0_vis_rgbdt_700demos_hcg_cue_nogate` | `data/hcue/L0_vis_700_nogate`: the same per-sensor networks and maps; the three distributions are averaged with no reliability estimate, the gate entries of the cue are 1 | does the gate that chooses among the sensors matter for the policy? |
| `L0_vis_rgbdt_700demos_hcg_cue_mapgate` | `data/hcue/L0_vis_700_v0`: the same maps; the gate is fitted on four numbers of every map and the peak distances (fusion mixture), it does not see the image | does the gate have to look at the image? |
| `L0_vis_rgbdt_700demos_hcg_cue_early` | `data/hcue/L0_vis_700_early`: one network of the same form on the five channels of the three sensors, no gate, gate entries 1 | are maps per sensor needed, or does one network sort the sensors out by itself? |
| `L0_vis_rgbdt_700demos_dp_aux` | none. The plain rgbdt policy; a head on the encoder's features predicts the hindsight grasp point of the episode and its error is added to the loss with weight 1 (`policy/train_auxgrasp.sh`, `policy/aux_grasp_policy.py`). At evaluation it sees what the plain baseline sees | is a cue in the conditioning needed, or is the supervision enough? |

**The cues of the controls, before any policy** (training demos out of fold / held-out demos; made with
`python policy/hindsight_cue.py <demos> --eval <held-out demos> --out <folder> --mode nogate | early | label`):

| cue | within 1.5 px | within 1 cm | within 1.5 cm | record |
|---|---|---|---|---|
| the method (verified gate, consensus) | 0.964 / 0.966 | 0.827 / 0.881 | 0.916 / 0.943 | [`perception/verified_gate.txt`](perception/verified_gate.txt) |
| no gate, mean of the three maps | 0.836 / 0.851 | 0.567 / 0.583 | 0.704 / 0.717 | [`perception/no_gate.txt`](perception/no_gate.txt) |
| no gate, product of the three maps (not run) | 0.696 / 0.673 | 0.341 / 0.363 | 0.456 / 0.479 | same |
| gate on map numbers, mixture | 0.900 / 0.903 | 0.730 / 0.786 | 0.830 / 0.856 | [`perception/gate_on_map_numbers.txt`](perception/gate_on_map_numbers.txt) |
| one network on the five channels | 0.95 / 0.97 | 0.82 / 0.83 | 0.92 / 0.93 | [`perception/one_network.txt`](perception/one_network.txt) |

Of the two forms without a gate the mean is run: it is the better one on the number the forms were chosen on (within
1 cm, training demos out of fold). The gate on map numbers is the folder kept from 11/22 (mixture; product and pick are
within 0.015 of it). The three sensors trained together (0.693 within 1.5 px) are below the form without a gate and
are not run as a policy. On the cue alone the one network on the five channels is as accurate as the method (within
1 cm 0.82 against 0.83 on the training demos out of fold): among the forms that combine maps per sensor the verified
gate is what makes the cue right (0.83 against 0.57 without a gate and 0.73 with a gate on map numbers), but a single
network on all sensors does not need it on these demos. This was seen before the policy runs and is why item 1 is
read in both directions. In the evaluation env (`policy/check_cue_env.py`, `j_cells11` / `k_cells11_heldout`, within
1.5 cm of the glass): the method 0.93 / 0.88, no gate 0.60 / 0.62, gate on map numbers 0.80 / 0.76, one network
0.90 / 0.87 ([`perception/env_check_controls.txt`](perception/env_check_controls.txt)).

**Read-out, fixed now.** "Same episodes" = the same test set and episode seed; sign test on the episodes in which
exactly one of the two succeeds; over all 1320 episodes and per cell; p < 0.05
(`python policy/readout_cue_controls.py`).

1. Each control against `L0_vis_rgbdt_700demos_hcg_cue`. If `hcg_cue` is above `hcg_cue_nogate`, the gate that
   chooses among the sensors contributes to the success of the policy; if it is above `hcg_cue_mapgate`, a gate that
   looks at the image contributes beyond one on map numbers; if it is above `hcg_cue_early`, maps per sensor with the
   gate are better than one network on all sensors, and if it is not, the structure per sensor is not shown to be
   needed for the success on this benchmark; if it is above `dp_aux`, the cue in the conditioning is needed
   beyond the supervision. A control that is not below `hcg_cue` with p < 0.05 means the corresponding part is not
   shown to matter, and is reported as such.
2. Each control against the rgbdt ImageNet baseline of seed 42 over all episodes, and per cell against the seed-42
   run of the best baseline of the cell (as in round 1). For `dp_aux` this says whether the auxiliary target alone
   helps the plain policy.
3. For the three controls with a cue, `python policy/analyze_cue.py <run>`: the share of episodes with a cue within
   1.5 cm of the glass and the success with a right and with a wrong cue. It separates a control that fails because
   its cue is wrong more often from one that fails with the same cue.
4. One seed each: a first reading. Seeds for the controls are decided after this round; the claim of Stage 6 is made
   for the method only, with the three seeds of item 5 of round 1.

The gate entries of the cue are 1 for `nogate` and `early`: the policy is not told how far a sensor is trusted, since
these forms have no such estimate. The weight of the auxiliary loss (1) and the head (one hidden layer of 256) are set
here and not tuned.

Queue: `data/runs/fv_queue.sh` of the worktree of [`../fusion_vis/README.md`](../fusion_vis/README.md) (one run per
GPU, three at once over both pages), in the order of the table, ahead of the runs of that page that have not started.
The seed runs of item 5 of round 1 are put ahead of the controls not yet started once round 1 has been read. A run
that fails is not started again.

### Results of round 2 as the controls finish

Full tables: [`readout_round2.txt`](readout_round2.txt) (`python policy/readout_cue_controls.py`) and
[`cue_in_episodes_round2.txt`](cue_in_episodes_round2.txt) (`python policy/analyze_cue.py`). Seed 42, 1320 episodes
per run. The read-out of the whole round is written when all four controls have finished.

**`hcg_cue_early`, one network on the five channels (11/23 17:20 KST).**

| cell | sensors that show the glass | `hcg_cue` (the method's cue) | `hcg_cue_early` | rgbdt ImageNet, seed 42 |
|---|---|---|---|---|
| 0 | rgb+depth+thermal | 0.950 | 0.883 | 0.775 |
| 1 | depth+thermal | 0.858 | 0.825 | 0.750 |
| 2 | rgb+thermal | 0.900 | 0.825 | 0.225 |
| 3 | rgb+depth | 0.933 | 0.858 (below) | 0.650 |
| 4 | thermal | 0.642 | 0.700 | 0.192 |
| 5 | depth | 0.792 | 0.742 | 0.650 |
| 6 | rgb | 0.883 | 0.742 (below) | 0.158 |
| 7 | rgb (faint) | 0.842 | 0.708 (below) | 0.092 |
| 8 | depth (faint) | 0.625 | 0.500 (below) | 0.525 |
| 9 | thermal (faint) | 0.467 | 0.492 | 0.050 |
| 10 | all faint | 0.858 | 0.750 (below) | 0.483 |
| macro average | | 0.795 | 0.730 (below) | 0.414 |

"(below)": below `hcg_cue` on the same episodes, p < 0.05.

1. Against `hcg_cue`: 0.730 against 0.795 over all episodes (only `hcg_cue` succeeds 215, only the control 128,
   p = 3.1e-06); below in five cells (3, 6, 7, 8, 10), above in none (cells 4 and 9, thermal alone, are 0.058 and
   0.025 higher, p >= 0.05). By read-out 1: maps per sensor with the gate are better than one network on all sensors,
   for the success of the policy and on the combinations of the training set.
2. Against the baselines: above the rgbdt ImageNet baseline of seed 42 (0.414; 508 / 91, p = 3.6e-71); per cell above
   the seed-42 run of the best baseline in eight cells (not 1, 5, 8), below in none. Most of the gain over the
   baselines does not need the structure per sensor: the control keeps 0.32 of the method's 0.38.
3. The cue inside the episodes: within 1.5 cm of the glass in 0.883 of the episodes (the method 0.907), median error
   0.61 cm (0.47); success with a right cue 0.800 (0.855), with a wrong cue 0.195 (0.220). The control does not fail
   mainly through more wrong cues: it succeeds less often with a cue that counts as right.

Outside the fixed read-out, on the same files: success falls with the error of the cue inside the 1.5 cm too (the
method's run: 0.93 below 0.5 cm, 0.80 between 0.5 and 1 cm, 0.57 between 1 and 1.5 cm), and the control's cue is less
exact (errors below 0.5 cm: 517 episodes against 704). Giving the control the method's distribution of cue errors
(seven bins) would put it at 0.761, so about half of the 0.065 is the less exact cue. The other half is there at the
same error: on the 335 episodes in which both cues are within 0.5 cm, 0.967 against 0.910 (26 / 7, p = 0.0013). What
differs there is that the method's cue tells the policy which sensors were trusted (the gate entries; 1 for the
control) and the two training runs themselves (one seed each); the run cannot separate these two.

**`hcg_cue_nogate`, the same maps per sensor averaged with no gate (11/23 21:31 KST).**

| cell | sensors that show the glass | `hcg_cue` | `hcg_cue_nogate` | cue within 1.5 cm of the glass, `hcg_cue` / `hcg_cue_nogate` | rgbdt ImageNet, seed 42 |
|---|---|---|---|---|---|
| 0 | rgb+depth+thermal | 0.950 | 0.858 (below) | 0.99 / 0.99 | 0.775 |
| 1 | depth+thermal | 0.858 | 0.733 (below) | 0.93 / 0.83 | 0.750 |
| 2 | rgb+thermal | 0.900 | 0.583 (below) | 0.96 / 0.92 | 0.225 |
| 3 | rgb+depth | 0.933 | 0.800 (below) | 0.98 / 0.93 | 0.650 |
| 4 | thermal | 0.642 | 0.267 (below) | 0.88 / 0.34 | 0.192 |
| 5 | depth | 0.792 | 0.467 (below) | 0.90 / 0.38 | 0.650 |
| 6 | rgb | 0.883 | 0.267 (below) | 0.97 / 0.47 | 0.158 |
| 7 | rgb (faint) | 0.842 | 0.292 (below) | 0.94 / 0.42 | 0.092 |
| 8 | depth (faint) | 0.625 | 0.308 (below) | 0.80 / 0.28 | 0.525 |
| 9 | thermal (faint) | 0.467 | 0.233 (below) | 0.68 / 0.29 | 0.050 |
| 10 | all faint | 0.858 | 0.708 (below) | 0.93 / 0.90 | 0.483 |
| macro average | | 0.795 | 0.502 (below) | 0.91 / 0.61 | 0.414 |

1. Against `hcg_cue`: 0.502 against 0.795 over all episodes (only `hcg_cue` succeeds 446, only the control 58,
   p = 3.4e-75), below in all eleven cells. By read-out 1: the gate that chooses among the sensors contributes to the
   success of the policy. The loss is largest where one sensor alone shows the glass (cells 4 to 9: 0.306 against
   0.708) and is there, smaller, where two or three do (cells 0 to 3 and 10: 0.737 against 0.900).
2. Against the baselines: above the rgbdt ImageNet baseline of seed 42 over all episodes (0.502 against 0.414,
   286 / 170, p = 6.2e-08), but per cell below the seed-42 run of the best baseline in cells 5 and 8 (depth alone:
   0.467 against 0.633 and 0.308 against 0.525) and above it in four (2, 7, 9, 10). A cue without the gate is not
   "better in nearly every case" even with one seed.
3. The cue inside the episodes: within 1.5 cm of the glass in 0.61 of the episodes (the method 0.91), median error
   1.03 cm (0.47); in the cells where one sensor alone shows the glass 0.28 to 0.47. Success with a right cue 0.667
   (0.855), with a wrong cue 0.239 (0.220). This control fails mostly because its cue is wrong more often; a policy
   trained on such a cue also uses a right one less (0.667 against 0.855), as it cannot know when to rely on it.

The two controls so far, together: of the method's 0.38 over the rgbdt baseline, averaging the maps per sensor with no
gate keeps 0.09, one network on all sensors keeps 0.32. The gate is what makes maps per sensor usable at all; against
a single network that sees all sensors the structure per sensor with the gate adds 0.065 on the combinations of the
training set (round 3 asks about the others).

## Round 3: visibility combinations that are not in the training set (plan of 11/23 12:40 KST, before any of it is run; decision of the user)

**Why.** The eleven cells of `L0_vis` are in the training demos and in both test sets; `k_cells11_heldout` holds out
backgrounds, lighting and distractors, not the cells. A model can therefore learn the eleven cases one by one, and the
benchmark cannot tell that from judging every sensor on its own: on the cue alone, one network on all sensors is as
accurate as the maps per sensor with the verified gate (round 2). What a gate per sensor should add is that it keeps
working when the sensors fail in a combination that was not in the training set. Two experiments measure that. Both
are additions: the level `L0_vis`, its training set, its test sets and the frozen numbers of
[`../grid_bench/README.md`](../grid_bench/README.md) are not changed. One seed (42) each: a first reading.

### 3a. New combinations at test, no new training

Test sets `l_cells15_new` (in-distribution scene pools) and `m_cells15_new_heldout` (held-out backgrounds, lighting
and distractors), added to `sim/configs/manifest.json`: level `L0_vis` with the cells replaced by the 15 combinations
of shows (0) / blind (1) / faint (2) per sensor that are not among the eleven training cells and in which at least one
sensor shows the glass at least faintly (27 combinations - 11 training cells - all blind). 600 episodes per test set
(seeds 100000 to 100599), 40 per cell.

| cell | rgb / depth / thermal | | cell | rgb / depth / thermal |
|---|---|---|---|---|
| 0 | faint / shows / shows | | 8 | blind / shows / faint |
| 1 | shows / faint / shows | | 9 | faint / faint / shows |
| 2 | shows / shows / faint | | 10 | faint / shows / faint |
| 3 | faint / shows / blind | | 11 | shows / faint / faint |
| 4 | faint / blind / shows | | 12 | faint / faint / blind |
| 5 | shows / faint / blind | | 13 | faint / blind / faint |
| 6 | blind / faint / shows | | 14 | blind / faint / faint |
| 7 | shows / blind / faint | | | |

Step 1, the cue alone (`python policy/check_cue_env.py <cue.pt> --test_set <set> -n 600 --save <file>`): the cue of
the method (`data/hcue/L0_vis_700`), without a gate, with the gate on map numbers and from one network (the cue
folders of round 2), all fitted on the 700 demos. Reported: the share of episodes with the cue within 1.5 cm of the
glass, per cell and over the 1200 episodes; the method against each of the three on the same episodes (sign test on
the episodes in which exactly one of the two cues is within 1.5 cm).

Step 2, policies trained on the 700 demos, their last checkpoints evaluated on the two test sets
(`bash policy/eval_new_cells.sh <group>/<run> ...`): `hcg_cue`, `hcg_full`, `hcg_blind`, the rgbdt ImageNet baseline
of seed 42, and the four controls of round 2 once they have finished. It is run whatever step 1 shows. The runs of the
learned attention weighting are not evaluated here.

Read-out, fixed now (same episodes, sign test, p < 0.05, over the 1200 episodes and per cell):

1. `hcg_cue` against `hcg_cue_early`, in both directions. If `hcg_cue` is above, the maps per sensor with the gate
   carry over to new combinations better than one network on all sensors; if they do not differ, no advantage of the
   structure per sensor is shown on new combinations either.
2. `hcg_cue` against `hcg_cue_nogate` and `hcg_cue_mapgate`.
3. `hcg_full` against `hcg_cue`: whether gating the image features matters on new combinations.
4. Every policy against the rgbdt ImageNet baseline of seed 42.
5. For orientation, not a test: the success of every policy on the 15 new cells next to its success on the 11
   training cells.

### 3b. Cells left out of the training set

Training set `results/data/sim_demos_L0_vis_400_nosingle.zarr`: the demos of the 700 that are in cells 0, 1, 2, 3 and
10 (`python policy/subset_demos.py ... --cells 0 --cells 1 --cells 2 --cells 3 --cells 10`; 140 + 3 x 70 + 50 = 400
demos). In it at most one sensor is blind at a time, or all three are faint. Left out: cells 4 to 9, in which one
sensor alone shows the glass, fully or faintly, and the other two are blind at once. Every sensor still has all three
of its states in the training set (shows, blind, faint). Test sets: `j_cells11` and `k_cells11_heldout` as always
(1320 episodes: 600 in the cells of the training set, 720 in the cells left out).

Step 1, the cue alone, fitted on the 400 demos (`python policy/hindsight_cue.py <the 400 demos> --eval
results/data/sim_demos_L0_vis_700_seed700.zarr --out data/hcue/L0_vis_400_nosingle[_nogate | _early]`, modes verify,
nogate and early): reported per cell on the held-out demos of the 700-demo set's sibling (which has all eleven cells)
and in the evaluation env on the two test sets (`policy/check_cue_env.py --save`), the method against the one network
and against no gate on the same episodes, in the cells left out and in the cells of the training set.

Step 2, policies on the 400 demos (protocol of the benchmark: 101 epochs, last checkpoint, ImageNet encoders, 55
episodes of `j_cells11` during training, 660 episodes on each test set), run whatever step 1 shows:

| run | what it is |
|---|---|
| `L0_vis_rgbdt_400demos_nosingle_hcg_cue` | arm `cue` with the cue of the method fitted on the 400 demos |
| `L0_vis_rgbdt_400demos_nosingle_hcg_full` | arm `full` (cue and gated image features) with the same cue |
| `L0_vis_rgbdt_400demos_nosingle_hcg_cue_early` | arm `cue` with the cue of one network on all sensors fitted on the 400 demos |
| `L0_vis_rgbdt_400demos_nosingle_imagenet` | the plain rgbdt baseline (`policy/train.sh`, ImageNet encoders) on the 400 demos |

Read-out, fixed now (same episodes, sign test, p < 0.05; separately over the 720 episodes of the cells left out and
the 600 of the cells of the training set, and per cell):

1. `nosingle_hcg_cue` against `nosingle_hcg_cue_early`, in both directions, read as in 3a.
2. `nosingle_hcg_full` against `nosingle_hcg_cue`: whether gating the image features matters when two sensors are
   blind at once for the first time.
3. Each of the three against `nosingle_imagenet`.
4. What leaving the cells out costs: each of the four against its counterpart trained on the 700 demos (`hcg_cue`,
   `hcg_full`, `hcg_cue_early`, the rgbdt ImageNet baseline of seed 42), on the same episodes of cells 4 to 9.
5. `python policy/analyze_cue.py <run>` for the three runs with a cue.

### Results of step 1 of 3a and 3b: the cue alone (11/23 13:10 KST; no policy involved)

Records: [`perception/round3_cue_new_cells.txt`](perception/round3_cue_new_cells.txt),
[`perception/round3_cue_nosingle.txt`](perception/round3_cue_nosingle.txt) (`python policy/readout_round3.py cue-new |
cue-nosingle`), the fits without cells 4 to 9 in `perception/nosingle_*.txt`. Share of episodes with the cue within
1.5 cm of the glass in the evaluation env; "(below)": below the method on the same episodes with p < 0.05.

3a, the 15 new combinations, cues fitted on the 700 demos (1200 episodes):

| cells | method | one network | no gate | gate on map numbers |
|---|---|---|---|---|
| all 15 | 0.956 | 0.934 (below, 57 / 31, p = 0.0073) | 0.891 (below, p = 6.1e-16) | 0.932 (below, p = 0.003) |
| a sensor shows the glass fully (0-11) | 0.964 | 0.939 (below) | 0.906 (below) | 0.944 (below) |
| none does (12-14) | 0.925 | 0.917 | 0.829 (below) | 0.887 |

On the eleven training cells the same cues are right in 0.907, 0.883, 0.614 and 0.781 of the 1320 episodes. The new
combinations are not harder than the training cells for any of the cues: most of them have one more sensor that shows
the glass. The method is above the one network by 0.02; per cell only cell 7 (shows / blind / faint) differs.

3b, cues fitted on the 400 demos without cells 4 to 9 (1320 episodes):

| cell | rgb / depth / thermal | method | one network | no gate |
|---|---|---|---|---|
| 4 | blind / blind / shows | 0.708 | 0.958 (above) | 0.283 (below) |
| 5 | blind / shows / blind | 0.842 | 0.700 (below) | 0.500 (below) |
| 6 | shows / blind / blind | 0.967 | 0.725 (below) | 0.458 (below) |
| 7 | faint / blind / blind | 0.925 | 0.667 (below) | 0.483 (below) |
| 8 | blind / faint / blind | 0.633 | 0.408 (below) | 0.317 (below) |
| 9 | blind / blind / faint | 0.533 | 0.383 (below) | 0.192 (below) |
| cells left out (4-9) | | 0.768 | 0.640 (below, 174 / 82, p = 9e-09) | 0.372 (below, p = 6.3e-68) |
| cells of the training set (0-3, 10) | | 0.933 | 0.952 (p = 0.099) | 0.902 (below, p = 0.0079) |

Against the same cue fitted on all 700 demos, on the same episodes of the cells left out: the method 0.768 against
0.863 (p = 6.5e-10), the one network 0.640 against 0.839 (p = 1.3e-22), no gate 0.372 against 0.364.

- When two sensors are blind at once for the first time, the cue from maps per sensor with the verified gate stays
  right more often than the cue of one network on all sensors (0.77 against 0.64); the one network loses twice as much
  as the method from leaving the cells out (0.20 against 0.10).
- It is not uniform: in cell 4 (thermal alone) the one network is far above the method (0.96 against 0.71). There the
  verifier of the blind depth image is unsure (median 0.46; 0.27 when fitted on all 700 demos), with 70 demos in which
  depth is blind; not examined further.
- In the cells of the training set the two do not differ (0.93 against 0.95); in cell 1 the one network is above.
- These are cues, not policies. Step 2 of both parts says whether the success follows.

### Order of the queue (set 11/23 12:40 KST, before any run of rounds 2 and 3 has started)

`hcg_cue_early`, `hcg_cue_nogate`, the evaluation of 3a for the runs that exist, the four runs of 3b,
`hcg_cue_mapgate`, `dp_aux`, the evaluation of 3a for the controls, then the runs of
[`../fusion_vis/README.md`](../fusion_vis/README.md) that have not started; the seed runs of item 5 of round 1 go to
the front once round 1 has been read. This replaces the order given in round 2 (the control with one network first:
it is the one the claim about the structure per sensor depends on most).

Change of 11/23 17:10 KST (order only, no run has changed): the evaluation of 3a for the controls is the last line of
the queue and waits until `hcg_cue_mapgate` and `dp_aux` have ended. The queue starts a line as soon as a GPU is free,
so right behind `dp_aux` it would have started while these two were still training and skipped them.

## Files

- `policy/hindsight_cue.py` (cue, verifier, fusion; fit and report), `policy/check_cue_env.py` (the cue in the env),
  `policy/cue_gate_encoder.py` (gated encoder), `policy/config/train_glass_cuegate_unet_image.yaml`,
  `policy/config/task/glass_pick_hcue.yaml`, `policy/train_cuegate.sh` (arms), `policy/analyze_cue.py` (read-out 3).
- `policy/glass_dataset.py`, `policy/glass_gym_env.py`: the `hcue` observation (the cue of the episode); nothing changes
  for a policy without that key.
- `perception/`: the reports of this page. Not in git: `data/hcue/` (networks, maps, cue files).
- Round 2: `policy/hindsight_cue.py --mode nogate | early | label` (the cues of the controls),
  `policy/aux_grasp_policy.py`, `policy/config/train_glass_auxgrasp_unet_image.yaml`, `policy/train_auxgrasp.sh` (the
  control with the auxiliary loss), `policy/readout_cue_gate.py` (read-out of round 1, items 1, 2 and 4),
  `policy/readout_cue_controls.py` (read-out of round 2).
- Round 3: `sim/configs/manifest.json` (test sets `l_cells15_new`, `m_cells15_new_heldout`),
  `policy/eval_new_cells.sh` (the policies on them), `policy/check_cue_env.py --save` (the cue per episode),
  `policy/readout_round3.py` (read-out).
