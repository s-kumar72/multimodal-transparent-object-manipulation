# Fusion methods on L0_bench

The two fusion methods of the project, applied to the benchmark of `results/hard_scenario/benchmark/` (`L0_bench`,
confirmed 11/15: rgb 0.38, rgbd 0.625, rgbdt 0.605) and compared with its plain Diffusion Policy runs, followed by four
ablations of the cue-gating run. Run on the `exp/fusion-bench` branch (exp/demo-scale + `policy/cue-gating` +
`policy/learned-attention-weighting`), 11/15 17:18 – 11/16 07:00 KST.

**Update, 11/16 07:55 – 18:40 KST (rounds 3–5, last part of this file):** a gate that switches a sensor on only where
the glass is visible in it (`vis_route`) reaches 0.694 over 800 episodes, above rgb, rgbt, rgbdt (0.626, p 0.005) and
rgbd (0.640, p 0.026); with sensor guidance at inference (×1.5, same checkpoints) it reaches 0.743 (p 1e-5 against
rgbd, 7e-7 against rgbdt), level with rgbd on normal and crossover episodes and far above it on see-through ones.
Sections up to "Next steps (proposals, not run)" are the state of 11/16 07:00 and are kept as written.

**Update, 11/16 21:26 – 11/17 04:08 KST (round 6, before "Files"):** the guidance scale chosen on validation episodes
is 3.0 (the largest of the grid); with it `vis_route` reaches 0.780 over three seeds (1200 episodes; as trained 0.698).
A control trained the same way without the routing (`nogate_rgbonly`, one seed) reaches 0.682 as trained, not
distinguishable from `vis_route` at the same seed (0.705), and gains nothing from the same guidance (0.680, 0.682).
So guidance needs the routing, and without guidance the routing is not shown to add to its training scheme: conclusion
11 of rounds 3–5 is narrowed by conclusion 17. The rounds 3–5 section is kept as written.

**Update, 11/17 04:29 – 14:28 KST (round 7, before "Files"):** two more seeds of the control change the round 6
picture. Over three seeds each, `vis_route` is above its control without guidance (0.698 vs 0.657, p 0.036; weak at the
seed level), and guidance at 3.0 gains +8.3 points with the routing against +2.5 without it (difference +5.8, 95%
+2.3 to +9.1), so conclusions 16–18 of round 6 are revised by conclusions 21–23. The scale 3.0 is an interior optimum
on the validation episodes. Why guidance gains more with the routing is answered only in part. The round 6 section is
kept as written.

**Update, 11/17 13:59 – 23:19 KST (round 8, before "Files"):** with a third seed of the baselines (rgbd 0.624, rgbdt
0.613 over 1200 episodes each) `vis_route` stays above both, as trained (0.698) and with guidance (0.780). Wrong gate
decisions injected at inference: small errors cost nothing, the costly error is a sensor left on where it does not
show the glass, and switching thermal off in every step changes nothing measurable, so a benefit of the thermal sensor
is not shown. A composition of separately trained rgbd and rgbt policies reaches 0.646, level with rgbd and rgbdt.

## Setup

- **Protocol** as in the benchmark: `L0_bench`, 500 demos (`results/data/sim_demos_L0_bench_500.zarr`), 101 epochs,
  the last checkpoint (epoch 100) evaluated on (a) `a_in_distribution` and (g) `g_heldout_all` with 50 episodes each
  (seeds 100000–100049), training seeds 42 and 43 (`scripts/run_experiment.sh` with `TRAIN=...`).
- **Re-evaluation with 4× the episodes**: every checkpoint, the benchmark baselines included, on (a) and (g) with 200
  episodes each (seeds 100000–100199) → `n200/<run>/`; 800 episodes per method over two seeds.
- **Episode groups**: *normal* = the depth sensor sees the glass; *see-through* = it sees through it
  (`glass_seethrough`, 25% of episodes); *crossover* = glass within ±1 °C of ambient (`is_crossover`, 30%). Of 200
  episodes, 156 / 44 / 24; of 800, 620 / 180 / 84 (crossover overlaps both).
- **Tests**: two-sided Fisher exact test against rgbdt on the same seeds; 95% Wilson intervals.

| run tag | method | how |
|---|---|---|
| `cuegate`, `cuegate_seed43` | cue gating, run D of `results/cue_gating/README.md`: cue-driven sigmoid gate, w in the conditioning, targeted modality dropout, thermal warm start, unimodal aux heads, online corruption, shared crop, ImageNet init | `policy/train_gated.sh gate rgbdt` |
| `fusion`, `fusion_seed43` | learned attention weighting (`fusion/`): DP + `FusionObsEncoder` initialized from the segmentation run `mix_ws` | `scripts/train_fusion_policy.sh rgbdt ... policy.obs_encoder.pretrained=<mix_ws/best.pt>` |
| `static` | ablation B: the same encoder, data and corruption, no gate (seed 42) | `train_gated.sh static` |
| `gate_nohole` | the gate with the depth-hole corruption off (seed 42) | `task.dataset.augment.depth_hole_p=0` |
| `gate_cue2` | the gate + a 7th cue `depth_object` + a depth warm start toward `1 - ep_glass_seethrough` with the thermal one's weight (0.1) and schedule (seed 42) | `task.shape_meta.obs.cue.shape=[7] +task.shape_meta.obs.cue.depth_object_ref=policy/config/depth_ref_front_cam.npy policy.gate_depth_target=seethrough` |
| `static_cue2` | `static_cue` (cue vector in the conditioning, no gate) with the 7-entry cue (seed 42) | `train_gated.sh static_cue` + the cue overrides |

The segmentation pre-training `mix_ws` (`fusion/train.sh`, stages seg + summary, L0_bench sim frames + real data) is
summarized in `seg_summary.md`: IoU 87.1 / 84.0 / 86.0 on sim sets a / f (crossover) / g, 70.3 on transpose and 44.3 on
trobi; its own thermal gate is 0.29 on set a and 0.05 on the crossover set f.

## Failed and repeated runs

No run was lost; each was evaluated on its epoch-100 checkpoint.

- **CUDA OOM at start-up** (other users' jobs on the GPUs), retried before any checkpoint: `cuegate_seed43` once,
  `static` once.
- **`fusion` (seed 42)** was killed by the kernel OOM killer at about epoch 85 (01:52 KST: three 200-episode evals and six
  trainings at once used all host RAM) and relaunched from `latest.ckpt` = epoch 75, so epoch 75 was trained twice. Diffusion
  Policy runs `num_epochs` again after a resume, so the run went past epoch 100; it was stopped at epoch 105 (04:17), and
  `latest.ckpt` was checked byte-identical to the epoch-100 top-k save before evaluation. Its `train_logs.json.txt` keeps
  the repeated epoch 75 and epochs 101–105 (not used).
- **`fusion_seed43`**: `run_experiment.sh` exited 127 right after training because the script was edited while it ran;
  the training was complete and the retry only ran the evaluation.
- **200-episode re-evaluations**: the first three attempts were stopped for the RAM problem above and rerun from scratch.

## Results: 50 episodes per test set (the benchmark protocol)

200 episodes per method (two seeds); p against rgbdt.

| method | seed 42 (a)/(g) | seed 43 (a)/(g) | all | normal (156) | see-through (44) | crossover (24) |
|---|---|---|---|---|---|---|
| rgb | 0.40/0.34 | 0.44/0.34 | 76/200 = 0.380 (0.32–0.45) | 0.372 | 0.409 | 0.333 |
| rgbd | 0.60/0.64 | 0.66/0.60 | 125/200 = 0.625 (0.56–0.69) | 0.788 | 0.045 | 0.750 |
| rgbt | 0.56/0.66 | 0.42/0.42 | 103/200 = 0.515 (0.45–0.58) | 0.506 | 0.545 | 0.542 |
| rgbdt (baseline) | 0.58/0.54 | 0.70/0.60 | 121/200 = 0.605 (0.54–0.67) | 0.712 | 0.227 | 0.417 |
| cue gating | 0.56/0.54 | 0.68/0.60 | 119/200 = 0.595 (0.53–0.66), p 0.92 | 0.571, p 0.013 | 0.682, p 4e-5 | 0.417, p 1 |
| learned attention | 0.50/0.62 | 0.36/0.50 | 99/200 = 0.495 (0.43–0.56), p 0.035 | 0.545, p 0.003 | 0.318, p 0.47 | 0.625, p 0.25 |

Ablations (seed 42, 100 episodes; rgbdt seed 42: 56/100, normal 0.64, see-through 0.27, crossover 0.42):
static 0.57 (normal 0.58, see-through 0.55, crossover 0.50); gate_nohole 0.62 (0.65, 0.50, 0.42); gate_cue2 0.54
(0.56, 0.45, 0.33); static_cue2 0.54 (0.54, 0.55, 0.33).

## Results: 200 episodes per test set

800 episodes per method (two seeds; 400 per seed), p against rgbdt:

| method | seed 42 / 43 (of 400) | all | normal (620) | see-through (180) | crossover (84) |
|---|---|---|---|---|---|
| rgb | 161 / 150 | 0.389, p 2e-21 | 0.395 | 0.367, p 0.002 | 0.417, p 0.12 |
| rgbd | 261 / 251 | 0.640, p 0.60 | 0.798, p 0.036 | 0.094, p 0.003 | 0.774, p 0.003 |
| rgbt | 228 / 177 | 0.506, p 2e-6 | 0.508 | 0.500, p 1e-8 | 0.262, p 3e-4 |
| rgbdt (baseline) | 239 / 262 | 0.626 (0.59–0.66) | 0.747 | 0.211 | 0.548 |
| cue gating | 239 / 252 | 0.614, p 0.64 | 0.623, p 3e-6 | 0.583, p 5e-13 | 0.298, p 0.002 |
| learned attention | 247 / 175 | 0.527, p 8e-5 | 0.571, p 8e-11 | 0.378, p 8e-4 | 0.512, p 0.76 |

Seed 42 only, 400 episodes each, p against rgbdt seed 42:

| run | all | normal (310) | see-through (90) | crossover (42) |
|---|---|---|---|---|
| rgbdt (baseline) | 0.598 | 0.706 | 0.222 | 0.571 |
| rgbd | 0.652, p 0.13 | 0.816, p 0.002 | 0.089, p 0.022 | 0.857, p 0.007 |
| cue gating | 0.598, p 1 | 0.603, p 0.009 | 0.578, p 2e-6 | 0.286, p 0.015 |
| learned attention | 0.618, p 0.61 | 0.652, p 0.17 | 0.500, p 2e-4 | 0.571, p 1 |
| static | 0.535, p 0.087 | 0.548, p 6e-5 | 0.489, p 3e-4 | 0.405, p 0.19 |
| gate_nohole | 0.625, p 0.47 | 0.665, p 0.30 | 0.489, p 3e-4 | 0.452, p 0.38 |
| gate_cue2 | 0.555, p 0.25 | 0.584, p 0.002 | 0.456, p 0.002 | 0.214, p 0.002 |
| static_cue2 | 0.585, p 0.77 | 0.587, p 0.002 | 0.578, p 2e-6 | 0.357, p 0.079 |

- Against cue gating (seed 42) no ablation differs significantly: static 0.535 (p 0.087), gate_nohole 0.625 (0.47),
  gate_cue2 0.555 (0.25), static_cue2 0.585 (0.77); gate_nohole normal 0.665 vs 0.603 (p 0.13), crossover 0.452 vs
  0.286 (p 0.17).
- Seed spread: learned attention 247 vs 175 of 400 (p 5e-7) and rgbt 228 vs 177 (p 4e-4); every other pair p ≥ 0.11.
- Ceiling of choosing: the best policy per (see-through, crossover) group gives 585/800 = 0.731 among the baselines (rgbd
  where depth sees the glass, rgbt where it does not) and 600/800 = 0.750 with the two methods (cue gating for
  see-through).
- Failures are mostly missed grasps. Median distance of the last gripper close to the glass in failed episodes: rgbdt
  4.2 cm, rgbd 3.5, cue gating 2.1, learned attention 2.5, gate_nohole 2.5 cm.

## Gate diagnostics

- **Cues** (cue gating seed 42, 800 first frames): `depth_rough` separates see-through perfectly (AUC 1.000, also 1.0 on
  the training demos); `depth_valid` is 1.0 in every frame (L0_bench depth has no holes); `thermal_contrast` separates
  crossover only weakly (AUC 0.69). The new `depth_object` cue separates see-through with AUC 0.79–0.90 on the training
  demos (distractors stand above the table too).
- **Weights** (mean over decision steps, share of rgb / depth / thermal, n=200 episodes):

| run | normal | see-through | AUC w_depth (normal > see-through) | AUC w_thermal (no crossover > crossover) |
|---|---|---|---|---|
| cue gating seed 42 | 0.44 / 0.14 / 0.42 | 0.44 / 0.14 / 0.42 | 0.48 | 0.47 |
| cue gating seed 43 | 0.44 / 0.14 / 0.42 | 0.44 / 0.14 / 0.42 | 0.51 | 0.50 |
| gate_nohole | 0.35 / 0.31 / 0.34 | 0.38 / 0.23 / 0.39 | 0.95 | 0.55 |
| gate_cue2 | 0.42 / 0.16 / 0.42 | 0.43 / 0.13 / 0.44 | 0.79 | 0.47 |

- **During training** (cue gating seed 42): the weights fall from 0.62 / 0.45 / 0.45 (step 500) to 0.05 / 0.02 / 0.05 at
  step 20000 and 0.03 / 0.01 / 0.03 at the end, while the thermal warm start (λ 0.1 → 0 over 20000 steps, target mean
  0.92) is on; its loss rises from 0.22 to 0.80. `w_thermal` does not follow |thermal gap| (Spearman −0.01, against 0.30
  for the cue). The depth warm start of gate_cue2 loses the same way (training logs: `w_depth` 0.42 → 0.14 by step
  8000, equal on see-through and other samples).

## Conclusion

1. **No fusion run reaches the 0.8–0.9 target, and none beats the baseline or the best single sensor.** rgbd alone is
   still the best overall (0.640 over 800 episodes).
2. **Cue gating trades one group for the others.** Over 800 episodes see-through rises from 0.21 to 0.58, normal falls
   from 0.75 to 0.62 and crossover from 0.55 to 0.30; overall 0.614 against 0.626. Its gate is constant (shares 0.44 /
   0.14 / 0.42 in every group): the policy uses little depth everywhere, which helps see-through and hurts the two groups
   where depth is the sensor to trust.
3. **The trade-off does not come from the gate.** `static` (same encoder and corruption, no gate) shows the same pattern
   (seed 42: see-through 0.49, normal 0.55); the gate adds 0.06 over it (p 0.087).
4. **The depth-hole corruption is what keeps the depth gate flat.** Without it (`gate_nohole`) the depth gate separates
   see-through (AUC 0.95; depth share 0.31 normal, 0.23 see-through) without any depth supervision, normal (0.665) is no
   longer significantly below the baseline (0.706, p 0.30) and see-through stays up (0.49 vs 0.22). Its overall 0.625 is
   the highest of the rgbdt runs at seed 42 but not significantly above rgbdt (0.598) or rgbd (0.652), and it is one seed.
5. **No thermal gate reacts to crossover** (AUC 0.47–0.55 in every gated run), and the thermal cue itself is weak (AUC
   0.69). So crossover stays below rgbd (0.86 at seed 42) in every rgbdt run, and is worse than the baseline where depth
   is also turned down (cue gating 0.29, gate_cue2 0.21).
6. **The depth warm start at the thermal one's weight does not form the gate** (gate_cue2), and the 7th cue does not
   help; with the cue in the conditioning and no gate (`static_cue2`) see-through is as high as with the gate (0.578).
7. **Learned attention weighting depends strongly on the seed**: seed 42 is level with the baseline (0.618 vs 0.598,
   see-through 0.50 vs 0.22), seed 43 is far below (0.438 vs 0.655), so the pooled 0.527 is below the baseline (p 8e-5).
   Its failures are missed grasps close to the glass (median 2.5 cm): it finds the glass but grasps imprecisely. The
   segmentation model's own thermal gate does drop on crossover (0.29 → 0.05).
8. **Choosing sensors cannot reach 0.8 by itself**: picking the best existing policy for each episode group gives 0.75.
   The rest has to come from grasp precision.

## Next steps (proposals of 11/16 07:00; 1 and 2 were then run, see rounds 3–5)

1. `gate_nohole` with seed 43, and `static` with the hole corruption off, to confirm conclusion 4 and separate it from the
   gate (2 runs, ~5 h on two GPUs).
2. A thermal gate that reacts to crossover: a thermal cue measured where depth or RGB sees an object (so warm distractors
   do not count), and a warm start strong enough to hold (weight 1.0 or no annealing), checked early with the training-log
   diagnostics `w_thermal_xo` / `w_thermal_notxo`.
3. Learned attention weighting with a third seed to settle the seed spread, and a look at its missed grasps (approach
   height and gripper timing).
4. Toward 0.8+: grasp precision (more demos or epochs, action horizon), since sensor choice is capped at about 0.75 here.

## Rounds 3–5: confirmation runs and the target-visibility gate (11/16 07:55 – 18:40 KST)

### Runs

All on `L0_bench`, rgbdt, 500 demos, 101 epochs, as above; n=50 and n=200 evaluations as above. Seed 42 unless noted.

| run tag | what | overrides on `policy/train_gated.sh gate rgbdt` |
|---|---|---|
| `gate_nohole_seed43` | second seed of `gate_nohole` | `task.dataset.augment.depth_hole_p=0 training.seed=43` |
| `static_nohole` | no gate, depth-hole corruption off (control of `gate_nohole`) | `train_gated.sh static` + `depth_hole_p=0` |
| `vis_route`, `vis_route_seed43` | **visibility routing**: no learned gate; depth / thermal are on (w = 1) when the glass is visible in them and off (w = 0, feature zeroed) otherwise; rgb always on; 10% of training samples rgb only; no gate loss; depth-hole corruption off | `VIS` below + `policy.obs_encoder.dropout_mode=none +policy.obs_encoder.route=visible +policy.obs_encoder.route_rgb_only_p=0.1 policy.gate_loss_weight=0` |
| `vis_soft` | the same two measurements give a continuous weight, `w = sigmoid(a (s − thr) / tau + b)` with `a` (init 1) and `b` (init 0) learned with the policy; everything else as `vis_route` | `VIS` + `+task.shape_meta.obs.cue.vis_raw=true ... +policy.obs_encoder.route=soft +policy.obs_encoder.route_tau=[5.69,1.82] +policy.obs_encoder.route_rgb_only_p=0.1 policy.gate_loss_weight=0` |
| `gate_vis` | the learned cue gate of `gate_nohole` (targeted dropout, warm start, aux heads unchanged) whose input also has the four visibility entries, and `gate_norm=max` (weights divided by the largest of the frame) | `VIS` + `+policy.obs_encoder.gate_norm=max` |

`VIS` = `task.dataset.augment.depth_hole_p=0 task.shape_meta.obs.cue.shape=[10] +task.shape_meta.obs.cue.vis_thr=[15,30]
+task.shape_meta.obs.cue.glass_mask=<masks .npy> +task.shape_meta.obs.cue.seg_ckpt=<mix_ws/best.pt>`.

Evaluation-only variants of the two `vis_route` checkpoints (n=200 per test set, `n200/<run>__<variant>/`):
`all_on` (`eval.py --route all_on`: no routing), `guide15` (`--guidance 1.5`), and for seed 42 also `latch`
(`--route_latch`: the first decision of the episode is kept) and `depth_first` (`--route depth_first`: thermal only when
depth does not show the glass). The guidance scale 1.5 was fixed before any guided evaluation; no other value was run.

### Target-visibility cues

The glass mask comes from the segmentation network of `fusion/` (`mix_ws/best.pt`, rgb + depth + thermal input); the
policy never sees the mask or the network's features. At the mask the cue vector gets four more entries
(`policy/obs_util.py visibility_cues`): **depth step** = median depth of a 5-pixel ring around the mask minus the median
inside it (8-bit units), **thermal step** = |median inside − median of the ring| / 2.73 raw counts, and the two flags
`step > threshold`. With no mask in the frame both flags are 1. Training reads masks precomputed over the demo zarr
(`policy/make_glass_masks.py`, 31120 frames, none empty); the eval env runs the network on CPU per frame.

- Thresholds 15 (depth) and 30 (thermal) were set on the training demos only: depth step ≥ 23 in every normal frame and
  ≤ 9 in every see-through frame; thermal step 1st percentile 40 without crossover, 99th percentile 26 with it.
- Held-out check before training, on frames of test sets (a) and (g) (100 episodes, 1098 frames, never used to train
  the segmentation network): both steps separate their failure with AUC 1.00 (first frames and all frames); the old
  frame-level `thermal_contrast` cue gives 0.65–0.69. The mask was found in every frame (IoU 0.85 normal, 0.83
  crossover, 0.79 see-through, 0.76 both).
- `tau` of `vis_soft` (5.69, 1.82) was fixed before training from the same demo statistics: at `a = 1, b = 0` the nearer
  of {1st percentile of visible frames, 99th percentile of not-visible frames} gets w = 0.9 / 0.1.

### Interrupted and resumed runs

Every run was evaluated on its epoch-100 checkpoint (checkpoints are written every 25 epochs; `latest.ckpt` times checked).

- **`gate_nohole_seed43`** was killed at epoch 57 (09:58) by the overshoot watchdog, which read "57" twice from two log
  lines as 5757. It resumed from the epoch-50 checkpoint, so epoch 50 was trained twice (`train_logs.json.txt` keeps
  epochs 50–57 twice). The watchdog now takes the last match only and needs two readings (`watch_overshoot5.sh`).
- **`vis_route` (seed 42)** lost a DataLoader worker to the kernel OOM killer at epoch 76 (12:14: five trainings at once)
  and resumed from the epoch-75 checkpoint; epoch 75 was trained twice.
- **`vis_soft`, `gate_vis`**: their first attempts (27 minutes) were killed by the same OOM event and restarted from
  scratch (12:25, 12:44); nothing of the first attempts was kept.
- Resumed runs were stopped by the watchdog when the log reached epoch 101 (Diffusion Policy trains `num_epochs` more
  epochs after a resume); epoch 101 was not finished and is not used.

### Results: 50 episodes per test set

| run | (a)/(g) | all | normal | see-through | crossover |
|---|---|---|---|---|---|
| rgbd, both seeds | 0.60/0.64, 0.66/0.60 | 125/200 = 0.625 | 0.788 | 0.045 | 0.750 |
| rgbdt, both seeds | 0.58/0.54, 0.70/0.60 | 121/200 = 0.605 | 0.712 | 0.227 | 0.417 |
| gate_nohole, both seeds | 0.64/0.60, 0.72/0.60 | 128/200 = 0.640 | 0.667 | 0.545 | 0.375 |
| static_nohole | 0.58/0.56 | 57/100 = 0.570 | 0.603 | 0.455 | 0.417 |
| **vis_route, both seeds** | 0.72/0.66, 0.76/0.66 | 140/200 = 0.700, p 0.14 vs rgbd, 0.059 vs rgbdt | 0.705 | 0.682 | 0.667 |
| vis_soft | 0.68/0.66 | 67/100 = 0.670 | 0.692 | 0.591 | 0.583 |
| gate_vis | 0.70/0.68 | 69/100 = 0.690 | 0.705 | 0.636 | 0.583 |

### Results: 200 episodes per test set, both seeds (800 episodes)

p: two-sided Fisher exact test on all episodes against rgbd / rgbdt, not corrected for the number of comparisons.

| method | seed 42 / 43 (of 400) | all | normal (620) | see-through (180) | crossover (84) | p vs rgbd / rgbdt |
|---|---|---|---|---|---|---|
| rgb | 161 / 150 | 0.389 | 0.395 | 0.367 | 0.417 | |
| rgbd | 261 / 251 | 0.640 | 0.798 | 0.094 | 0.774 | – / 0.60 |
| rgbt | 228 / 177 | 0.506 | 0.508 | 0.500 | 0.262 | |
| rgbdt | 239 / 262 | 0.626 | 0.747 | 0.211 | 0.548 | 0.60 / – |
| cue gating | 239 / 252 | 0.614 | 0.623 | 0.583 | 0.298 | 0.30 / 0.64 |
| gate_nohole | 250 / 268 | 0.647 (0.61–0.68) | 0.690 | 0.500 | 0.452 | 0.79 / 0.41 |
| **vis_route** | 282 / 273 | **0.694** (0.66–0.72) | 0.734 | 0.556 | 0.655 | 0.026 / 0.005 |
| vis_route, `all_on` | 234 / 248 | 0.603 (0.57–0.64) | 0.721 | 0.194 | 0.464 | 0.13 / 0.36 |
| **vis_route, `guide15`** | 304 / 290 | **0.743** (0.71–0.77) | 0.792 | 0.572 | 0.774 | 1e-5 / 7e-7 |

- `vis_route` against the other methods: rgb p 1e-34, rgbt 2e-14, cue gating 9e-4, gate_nohole 0.055. Per group against
  rgbd: normal 0.734 vs 0.798 (p 0.009), see-through p 1e-21, crossover 0.655 vs 0.774 (p 0.12); against rgbdt: normal
  p 0.65, see-through p 2e-11, crossover 0.655 vs 0.548 (p 0.21); crossover against gate_nohole p 0.013.
- Per seed: `vis_route` 0.705 vs rgbd 0.652 (p 0.13) and rgbdt 0.598 (p 0.002) at seed 42; 0.682 vs 0.627 (p 0.12) and
  0.655 (p 0.45) at seed 43. `guide15` 0.760 vs 0.652 (p 0.001) and 0.598 (p 1e-6) at seed 42; 0.725 vs 0.627
  (p 0.004) and 0.655 (p 0.039) at seed 43.
- Same checkpoint, same episodes (sign test on the episodes whose outcome changed against `vis_route` as trained):
  `all_on` lost 104, gained 31 (p 2e-10; see-through lost 70, gained 5; crossover lost 26, gained 10; normal lost 34,
  gained 26). `guide15` gained 65, lost 26 (p 5e-5; normal gained 55, lost 19; see-through 10 / 7; crossover 16 / 6).
- Failure outcomes over 800 episodes: rgbd missed grasp 157, other 65, dropped 53, wrong height 13; rgbdt 152 / 49 / 60 /
  35; `vis_route` 111 / 9 / 80 / 44; `guide15` 80 / 14 / 72 / 38. Median distance of the last gripper close in failed
  episodes: 3.5, 4.2, 3.2 and 3.3 cm.

### Results: 200 episodes per test set, seed 42 (400 episodes)

| run | all | normal (310) | see-through (90) | crossover (42) | p vs vis_route |
|---|---|---|---|---|---|
| rgbd | 0.652 | 0.816 | 0.089 | 0.857 | 0.13 |
| rgbdt | 0.598 | 0.706 | 0.222 | 0.571 | 0.002 |
| gate_nohole | 0.625 | 0.665 | 0.489 | 0.452 | 0.020 |
| static_nohole | 0.573 | 0.577 | 0.556 | 0.429 | 1e-4 |
| gate_vis | 0.655 | 0.684 | 0.556 | 0.476 | 0.15 |
| vis_soft | 0.700 | 0.739 | 0.567 | 0.690 | 0.94 |
| **vis_route** | 0.705 | 0.739 | 0.589 | 0.690 | – |
| vis_route, `depth_first` | 0.705 | 0.745 | 0.567 | 0.738 | 1 (lost 20, gained 20) |
| vis_route, `latch` | 0.715 | 0.748 | 0.600 | 0.762 | 0.82 (lost 4, gained 8, sign test p 0.39) |
| vis_route, `all_on` | 0.585 | 0.703 | 0.178 | 0.381 | 5e-4 |
| vis_route, `guide15` | 0.760 | 0.800 | 0.622 | 0.762 | 0.093 (gained 33, lost 11, sign test p 0.001) |

`static_nohole` against `gate_nohole`: 0.573 vs 0.625 (p 0.15), normal 0.577 vs 0.665 (p 0.031).

### Gate behaviour

- **First decision of `vis_route`** (800 episodes): depth off in 180/180 see-through episodes and on in 620/620 others;
  thermal off in 82/84 crossover episodes and on in 716/716 others. In the training log (step 1000) the mean depth weight
  is 0.00 on see-through samples and 0.90 on the others; thermal 0.00 on crossover samples and 0.72 on the others (the
  rest is the 10% rgb-only samples and, for thermal, frames whose contrast the online corruption removed).
- **Switches during an episode**: in 57/800 episodes a flag changed after the first decision (first change: thermal off
  28, depth off 20, thermal on 5, depth on 4); 4 of them succeeded, against 551/743 = 0.742 without a change. They are
  a consequence of the failure, not its cause: of the 25 such episodes at seed 42, 2 succeeded as trained and 1 with the
  first decision latched (13 still dropped the glass, 6 missed it).
- **Weights at the first decision** (seed 42 unless noted, 400 episodes; rgb / depth / thermal):

| run | normal, no crossover | see-through | crossover | AUC depth (see-through) | AUC thermal (crossover) |
|---|---|---|---|---|---|
| gate_nohole | 0.051 / 0.047 / 0.049 | 0.056 / 0.035 / 0.059 | 0.052 / 0.046 / 0.050 | 0.99 | 0.54 |
| gate_nohole_seed43 | 0.065 / 0.058 / 0.063 | 0.075 / 0.039 / 0.073 | 0.066 / 0.056 / 0.063 | 1.00 | 0.62 |
| gate_vis | 1 / 0.729 / 0.409 | 1 / 0.678 / 0.486 | 1 / 0.721 / 0.396 | 1.00 | 0.81 |
| vis_soft | 1 / 0.942 / 0.989 | 1 / 0.293 / 0.936 | 1 / 0.853 / 0.271 | 1.00 | 0.98 |
| vis_route | 1 / 1 / 1 | 1 / 0 / 0.944 | 1 / 0.857 / 0.024 | 1.00 | 0.99 |

- **`vis_soft` learned a softer gate**: `a` fell from 1 to 0.42 (depth) and 0.12 (thermal), `b` rose to 0.31 and 0.63
  (0.72 / 0.74 and 0.19 / 0.17 at step 10000, 0.49 / 0.33 and 0.29 / 0.45 at step 20500). A sensor that does not show
  the glass keeps a weight instead of 0: depth 0.21–0.52 on see-through episodes (median 0.28), thermal 0.20–0.35 on
  crossover episodes (median 0.25; 1.0 in the one of 42 where the thermal step is above the threshold).
- **`gate_vis`**: the normalization removes the common shrinking (rgb is always 1) but the learned weights barely move
  with the visibility: depth 0.66–0.70 on see-through against 0.71–0.75 otherwise, thermal 0.40 on crossover against
  0.41 otherwise, from step 10000 to the end of training.

### Conclusions of rounds 3–5

9. **`gate_nohole` holds at a second seed and is level with the baselines**: 0.647 over 800 episodes (250 and 268 of
   400) against rgbd 0.640 and rgbdt 0.626. The pattern is the same at both seeds: see-through up (0.500), normal below
   rgbd (0.690 vs 0.798, p 2e-5) and rgbdt (0.747, p 0.032), crossover 0.452. Its weights stay near 0.05 and move
   little (depth share 0.30 → 0.21–0.23 on see-through), and thermal does not react to crossover (AUC 0.54–0.62).
   Without the gate (`static_nohole`) the same setup gives 0.573 (p 0.15 against `gate_nohole`).
10. **Switching sensors on the visibility of the glass beats all four baselines in overall success**: `vis_route` 0.694
    over 800 episodes, above rgbd and rgbdt at each seed. The margin over rgbd (p 0.026) does not survive a correction
    for four comparisons; the one over rgbdt (p 0.005) does. See-through is far above rgbd and rgbdt (0.556 vs 0.094 /
    0.211), crossover is above rgbdt and the learned gates (0.655 vs 0.548 / 0.452) and below rgbd (0.774, p 0.12),
    normal is level with rgbdt and below rgbd (0.734 vs 0.798, p 0.009).
11. **The gain is the routing decision.** With all sensors on, the same checkpoints fall to 0.603, i.e. to the plain
    rgbdt level, losing see-through (0.194) and crossover (0.464).
12. **The details of the rule do not matter here.** Keeping the first decision or preferring depth changes nothing
    (0.715, 0.705 against 0.705), and a continuous learned weight (`vis_soft`, 0.700) equals the 0 / 1 rule although
    training does move it away from 0 / 1. The two failures of this benchmark are binary, so this says nothing about
    partially visible targets.
13. **A learned gate still does not switch.** Given the visibility entries and normalized weights (`gate_vis`), the MLP
    gate separates the cases (AUC 1.00 / 0.81) but changes the weights by a few percent; the run is at 0.655 with
    crossover at 0.476. The runs at 0.70 are the two whose weight of an unseen sensor is about 0.3 or less (`vis_soft`) or 0 (`vis_route`).
14. **Sensor guidance at inference adds 5 points and closes the gap on normal episodes**: 0.743 over 800 episodes
    (0.760, 0.725), normal 0.792 (rgbd 0.798), crossover 0.774 (rgbd 0.774), see-through 0.572. It is above rgbd and
    rgbdt at each seed (weakest: p 0.039 against rgbdt at seed 43) and inside the ceiling of choosing between existing
    policies (0.73–0.75, above).

Caveats: two seeds; one guidance scale, fixed in advance, not tuned; the baselines were not evaluated with guidance
(they have no rgb-only mode), so the share of the 5 points that needs the gate is not separated; the segmentation
network that places the mask was trained on L0_bench frames of other episodes, and the eval env runs it on CPU (an
evaluation takes about twice as long).

### Related work checked (11/16)

Composition of sensor-specific diffusion policies at the score level exists: PoCo (RSS 2024), Modality-Composable
Diffusion Policy (arXiv 2503.12466, fixed weights) and a policy-consensus variant with router-learned weights that are
fixed at execution (arXiv 2509.23468; RGB / point cloud / tactile). Rejecting a corrupted sensor with a detector is in
Detect-Reject-Correct (ICRA 2021). What differs here: the on / off decision is measured per frame at the manipulated
object, in the physical units of each sensor, and the same decision is used in training, so every sensor subset is a
trained mode of one policy.

### Next steps (proposals of 11/16 18:40; 1–3 were then run, see round 6)

1. Separate guidance from the gate: an rgbdt policy trained with the 10% rgb-only samples and no routing, evaluated with
   the same guidance (1 training run + evaluations).
2. Guidance scale chosen on validation episodes with seeds outside the test range, and the same guidance on `vis_soft`.
3. A third seed of `vis_route`.
4. Partially visible targets (intermediate thermal gaps, partial see-through) to test 0 / 1 against continuous weights
   where they can differ; this needs new test conditions and is outside the fixed benchmark.
5. Learned attention weighting with a third seed, as proposed above.

## Round 6: guidance control, guidance scale, third seed (11/16 21:26 – 11/17 04:08 KST)

### Runs

All on `L0_bench`, rgbdt, 500 demos, 101 epochs, as above. No code changed in this round. Both trainings ran to epoch
100 without interruption; all 22 queued jobs finished at the first attempt.

| run tag | what | overrides on `policy/train_gated.sh gate rgbdt` |
|---|---|---|
| `nogate_rgbonly` (seed 42) | **control of `vis_route`**: the same code path and settings, but the visibility thresholds are −1000, so depth and thermal are never switched off (weights 1 / 1 / 1 in every evaluated step); the 10% rgb-only training samples stay, so the policy has the rgb-only mode that guidance needs | the `vis_route` overrides with `vis_thr=[-1000,-1000]` |
| `vis_route_seed44` | third seed of `vis_route` | the `vis_route` overrides + `training.seed=44` |

Evaluations (`eval.py`, 10 envs):

- **Validation**, `val100/<run>__g{100,125,150,200,300}/`: 100 episodes per test set with `--test_start_seed 200000`
  (demos use seeds 0–499, test episodes 100000–100199), `--guidance` 1.0, 1.25, 1.5, 2.0, 3.0, on the `vis_route`
  checkpoints of seeds 42 and 43.
- **Test**, `n200/<run>[__guide15|__guide30]/`: 200 episodes per test set on the test seeds, as trained and with
  `--guidance 1.5` / `3.0`, for `nogate_rgbonly`, `vis_route_seed44`, `vis_soft`, and `guide30` for the two earlier
  `vis_route` checkpoints.

### Guidance scale on validation episodes

Rule, fixed before these evaluations: the scale with the highest success pooled over both checkpoints and both test
sets (400 episodes per scale) is taken, ties go to the smaller scale; if it is not 1.5, test evaluations with it are added.

| scale | pooled (400) | normal (300) | see-through (100) | crossover (44) | seed 42 (a) / (g) | seed 43 (a) / (g) | same episodes against 1.0 |
|---|---|---|---|---|---|---|---|
| 1.0 (no guidance) | 284 = 0.710 | 0.753 | 0.580 | 0.705 | 79 / 70 | 75 / 60 | – |
| 1.25 | 298 = 0.745 | 0.790 | 0.610 | 0.773 | 83 / 70 | 82 / 63 | gained 22, lost 8 (p 0.016) |
| 1.5 | 298 = 0.745 | 0.793 | 0.600 | 0.727 | 85 / 70 | 82 / 61 | gained 25, lost 11 (p 0.029) |
| 2.0 | 309 = 0.773 | 0.823 | 0.620 | 0.773 | 85 / 74 | 84 / 66 | gained 36, lost 11 (p 3e-4) |
| **3.0** | **318 = 0.795** | 0.840 | 0.660 | 0.727 | 89 / 74 | 88 / 67 | gained 49, lost 15 (p 2e-5) |

**Selected: 3.0.** It is the largest scale of the grid and success rises over the whole grid, so the best scale may be
larger; the grid was not extended after seeing this. On these 400 episodes 3.0 against 1.5 is p 0.11 (Fisher).

### Results on the test seeds: `vis_route`, three seeds

200 episodes per test set and seed. The baselines have two seeds (800 episodes), `vis_route` three (1200). p: two-sided
Fisher exact test on all episodes against rgbd / rgbdt, not corrected for the number of comparisons.

| method | seed 42 / 43 / 44 (of 400) | all | normal | see-through | crossover | p vs rgbd / rgbdt |
|---|---|---|---|---|---|---|
| rgbd | 261 / 251 / – | 0.640 | 0.798 | 0.094 | 0.774 | |
| rgbdt | 239 / 262 / – | 0.626 | 0.747 | 0.211 | 0.548 | |
| vis_route, as trained | 282 / 273 / 282 | 0.698 (0.67–0.72) | 0.741 | 0.548 | 0.643 | 0.007 / 0.001 |
| vis_route, `guide15` | 304 / 290 / 295 | 0.741 (0.72–0.76) | 0.795 | 0.556 | 0.738 | 2e-6 / 6e-8 |
| **vis_route, `guide30`** (selected scale) | 311 / 305 / 320 | **0.780** (0.76–0.80) | 0.840 | 0.574 | 0.778 | 1e-11 / 1e-13 |

- Seeds 42 and 43 only (the seeds of the baselines, 800 episodes): `guide30` 0.770 (normal 0.827, see-through 0.572,
  crossover 0.762), p 1e-8 / 5e-10. Per seed against the same-seed baselines: 0.777 vs 0.652 (p 1e-4) and 0.598
  (p 5e-8) at seed 42; 0.762 vs 0.627 (p 4e-5) and 0.655 (p 0.001) at seed 43.
- Seed 44, whose checkpoint took no part in choosing the scale: 0.705 as trained, 0.738 with `guide15`, 0.800 with `guide30`.
- Same checkpoints, same episodes, against `vis_route` as trained (1200 episodes, sign test): `guide15` gained 93,
  lost 41 (p 8e-6); `guide30` gained 160, lost 61 (p 2e-11): normal gained 130, lost 38 (p 6e-13), crossover 24 / 7
  (p 0.003), see-through 30 / 23 (p 0.41). `guide30` against `guide15`: 936 vs 889 of 1200 (Fisher p 0.028; gained 112,
  lost 65, sign test p 5e-4).
- Per group, `guide30` against rgbd: normal 0.840 vs 0.798 (p 0.041), crossover 0.778 vs 0.774, see-through 0.574 vs
  0.094; against rgbdt: normal p 9e-6, crossover p 8e-4. As trained against rgbd: normal 0.741 vs 0.798 (p 0.010),
  crossover 0.643 vs 0.774 (p 0.048).
- Per test set: as trained (a) 0.732, (g) 0.663; `guide30` (a) 0.807, (g) 0.753; rgbd 0.658 / 0.623; rgbdt 0.640 / 0.613.
- Failure outcomes over 1200 episodes: as trained missed grasp 164, dropped 111, wrong height 69, other 18; `guide15`
  123 / 97 / 63 / 24; `guide30` 101 / 91 / 46 / 20.
- First decision at seed 44 as at the other seeds: depth off in 90/90 see-through episodes, on in 310/310 others;
  thermal off in 41/42 crossover episodes, on in 358/358 others; a flag changed later in 28 episodes.
- 50 episodes per test set (benchmark protocol): `vis_route_seed44` (a) 0.76, (g) 0.60; three seeds 208/300 = 0.693
  (normal 0.701, see-through 0.667, crossover 0.639).

### Results on the test seeds: control and `vis_soft`, seed 42 (400 episodes)

| run | guidance | all | normal (310) | see-through (90) | crossover (42) | same episodes against the run as trained |
|---|---|---|---|---|---|---|
| rgbd | | 261 = 0.652 | 0.816 | 0.089 | 0.857 | |
| rgbdt | | 239 = 0.598 | 0.706 | 0.222 | 0.571 | |
| **nogate_rgbonly** | – | 273 = 0.682 (0.64–0.73) | 0.710 | 0.589 | 0.500 | |
| nogate_rgbonly | 1.5 | 272 = 0.680 | 0.710 | 0.578 | 0.524 | gained 22, lost 23 (p 1) |
| nogate_rgbonly | 3.0 | 273 = 0.682 | 0.729 | 0.522 | 0.500 | gained 36, lost 36 (p 1) |
| vis_route | – | 282 = 0.705 (0.66–0.75) | 0.739 | 0.589 | 0.690 | |
| vis_route | 1.5 | 304 = 0.760 | 0.800 | 0.622 | 0.762 | gained 33, lost 11 (p 0.001) |
| vis_route | 3.0 | 311 = 0.777 | 0.829 | 0.600 | 0.786 | gained 52, lost 23 (p 0.001) |
| vis_soft | – | 280 = 0.700 | 0.739 | 0.567 | 0.690 | |
| vis_soft | 1.5 | 289 = 0.723 | 0.774 | 0.544 | 0.762 | gained 31, lost 22 (p 0.27) |
| vis_soft | 3.0 | 294 = 0.735 | 0.794 | 0.533 | 0.690 | gained 51, lost 37 (p 0.17) |

- **Control against `vis_route`, both as trained**: 0.682 vs 0.705, Fisher p 0.54; on the same episodes `vis_route`
  gained 35 and lost 26 (sign test p 0.31). See-through 53/90 in both; normal 220 vs 229 of 310 (p 0.47); crossover
  21 vs 29 of 42 (p 0.12).
- **Control against the other seed-42 runs, as trained**: rgbdt 0.598 (p 0.015; see-through 0.589 vs 0.222, p 9e-7;
  normal 0.710 vs 0.706; crossover 0.500 vs 0.571, p 0.66); rgbd 0.652 (p 0.41; normal 0.710 vs 0.816, p 0.002;
  crossover 0.500 vs 0.857, p 9e-4); `static_nohole` 0.573 (p 0.002; normal 0.710 vs 0.577, p 8e-4); `gate_nohole`
  0.625 (p 0.10); `gate_vis` 0.655 (p 0.45).
- **Gain from guidance, `vis_route` minus control** (bootstrap over the 400 shared test episodes): +5.8 points at 1.5
  (95% interval +1.3 to +10.2), +7.2 points at 3.0 (+1.8 to +12.8).
- **With guidance, `vis_route` against control**: 0.760 vs 0.680 at 1.5 (Fisher p 0.015; same episodes gained 52,
  lost 20, p 2e-4); 0.777 vs 0.682 at 3.0 (p 0.003; gained 67, lost 29, p 1e-4; normal 257 vs 226 of 310, crossover 33
  vs 21 of 42, see-through 54 vs 47 of 90).
- **`vis_soft`**: its gains (+2.3 and +3.5 points) are not significant; `vis_route`'s gain minus `vis_soft`'s is +3.3
  points (−1.5 to +8.0) at 1.5 and +3.7 (−1.3 to +8.7) at 3.0, and guided `vis_soft` against guided `vis_route` is
  p 0.26 / 0.19.
- Failure outcomes of the control (400 episodes): missed grasp 65, dropped 36, wrong height 17, other 5; with
  `guide30` 60 / 40 / 15 / 9.
- 50 episodes per test set: `nogate_rgbonly` (a) 0.68, (g) 0.56, 62/100 (normal 0.641, see-through 0.545, crossover 0.583).

### Conclusions of round 6

15. **The selected guidance scale is 3.0 and its gain holds on the test seeds**: 0.780 over three seeds against 0.698
    as trained, above rgbd and rgbdt at each seed, and 0.800 at the seed that took no part in the selection. Normal
    episodes rise to 0.840 (rgbd 0.798, p 0.041 uncorrected), crossover to the rgbd level (0.778 vs 0.774); see-through
    does not change with guidance (0.548 → 0.574). The scale is at the edge of the grid.
16. **Guidance needs the routing.** The control, trained the same way without the routing, gains nothing from the same
    guidance at either scale (0.682 → 0.680 → 0.682, as many episodes lost as gained), while `vis_route` gains 5.5 and
    7.2 points at the same seed; the difference of the gains excludes 0 at both scales. `vis_soft`, whose unseen-sensor
    weight is about 0.3 instead of 0, lies between (+2.3, +3.5, not significant).
17. **Without guidance the routing is not shown to add to its training scheme** (narrows conclusion 11). The control
    reaches 0.682 against 0.705, with the same see-through success (53/90); only crossover differs (0.500 vs 0.690 on
    42 episodes, p 0.12). The `all_on` test of conclusion 11 shows that a policy trained with the routing needs it at
    inference; it does not show that a policy trained without it is worse. The see-through gain over rgbdt (0.589 vs
    0.222) comes from the training scheme, not from switching depth off. This rests on one seed of the control, and
    400 episodes cannot resolve a difference under about 7 points.
18. **The control is the best policy without a gate here**: above rgbdt (0.598, p 0.015) and `static_nohole` (0.573,
    p 0.002), level with rgbd (0.652) with the opposite profile (see-through 0.589 vs 0.089, crossover 0.500 vs 0.857).
    It differs from `static_nohole` in the 10% rgb-only samples and in the encoder family (missing-sensor embedding,
    weights in the conditioning, unimodal aux heads); which of these matters is not separated.
19. **A third seed confirms `vis_route` as trained**: 282/400, three seeds 0.698 (p 0.007 against rgbd, 0.001 against
    rgbdt). It stays below rgbd on normal (0.741 vs 0.798, p 0.010) and crossover episodes (0.643 vs 0.774, p 0.048).

Caveats: the control has one seed; the scale was selected at the edge of its grid; the validation episodes have other
seeds than the test episodes but two of the three tested checkpoints are the ones the scale was chosen on (the third,
seed 44, is not); rgbd and rgbdt have two seeds and cannot be guided (no rgb-only mode), so the control is the only
guided comparison without a gate; p values are not corrected for the number of comparisons. Why guidance does nothing
without the routing was not examined: on normal episodes both policies have all sensors on, yet `vis_route` gains
there (seed 42: 42 gained, 14 lost at 3.0) and the control does not (29 / 23).

### Next steps (proposals of 11/17 04:10; 1–4 were then run, see round 7)

1. Seeds 43 and 44 of the control (2 trainings, 3 evaluations each) to settle conclusion 17 and repeat conclusion 16.
2. Validation at scales above 3.0 (4, 5, 7) on the same validation episodes, then the test evaluations of the selected
   scale (evaluations only).
3. Why guidance needs the routing: size and direction of the difference between the routed and the rgb-only noise
   prediction on the same frames, `vis_route` against the control (no training).
4. What lifts the control above `static_nohole`: the control without the rgb-only samples (`route_rgb_only_p=0`, 1 training).
5. Partially visible targets and the third seed of learned attention weighting, as proposed above.

## Round 7: control seeds, larger guidance scales, guidance probe, control without rgb-only samples (11/17 04:29 – 14:28 KST)

### Runs

All on `L0_bench`, rgbdt, 500 demos, 101 epochs, as above. The three trainings ran to epoch 100 without interruption;
all 16 queued jobs finished at the first attempt. No training or evaluation code changed; one analysis script was
added (`policy/analyze_guidance.py`).

| run tag | what | overrides on `policy/train_gated.sh gate rgbdt` |
|---|---|---|
| `nogate_rgbonly_seed43`, `nogate_rgbonly_seed44` | second and third seed of the control of `vis_route` (sensors never switched off, 10% rgb-only training samples) | the `nogate_rgbonly` overrides + `training.seed=43` / `44` |
| `nogate_allon` (seed 42) | the control without the rgb-only training samples: all three sensors on in every training sample (logged mean weights 1.0000). It has no rgb-only mode, so it cannot be guided | the `nogate_rgbonly` overrides with `route_rgb_only_p=0.0` |

Evaluations and analysis:

- **Validation**, `val100/<run>__g{400,500,700}/`: the validation episodes of round 6 with `--guidance` 4.0, 5.0, 7.0,
  on the `vis_route` checkpoints of seeds 42 and 43.
- **Test**, `n200/<run>[__guide15|__guide30]/`: 200 episodes per test set on the test seeds, as trained and with
  `--guidance 1.5` / `3.0` for the two new control seeds, as trained for `nogate_allon`.
- **Probe**, `guidance_probe/`: `policy/analyze_guidance.py` on the seed-42 checkpoints of `vis_route` and of the
  control (no simulator).

Fixed before the results: the scale rule (below); for the control, the three seeds are pooled (1200 episodes) and
compared with `vis_route` over its three seeds on (a) success as trained, (b) the gain from guidance at 1.5 and 3.0
within the control, (c) the difference of the two policies' gains (bootstrap over the test episodes); for the probe,
the two measures listed in its section. Everything else in this round is marked exploratory.

### Guidance scale above 3.0 (validation episodes)

Rule, fixed before these evaluations: the round 6 rule over all eight scales (highest success pooled over both
checkpoints and both test sets, ties to the smaller scale). If the selected scale is not 3.0, test evaluations with it
are added; if it is 7.0, the grid is not extended again.

| scale | pooled (400) | normal (300) | see-through (100) | crossover (44) | seed 42 (a) / (g) | seed 43 (a) / (g) | same episodes against 3.0 |
|---|---|---|---|---|---|---|---|
| 3.0 (round 6) | 318 = 0.795 | 0.840 | 0.660 | 0.727 | 89 / 74 | 88 / 67 | – |
| 4.0 | 307 = 0.768 | 0.830 | 0.580 | 0.727 | 83 / 70 | 84 / 70 | gained 8, lost 19 (p 0.052) |
| 5.0 | 303 = 0.758 | 0.813 | 0.590 | 0.705 | 80 / 75 | 83 / 65 | gained 10, lost 25 (p 0.017) |
| 7.0 | 303 = 0.758 | 0.820 | 0.570 | 0.750 | 80 / 72 | 80 / 71 | gained 16, lost 31 (p 0.040) |

**Selected: 3.0, unchanged**, now inside the grid (0.710, 0.745, 0.745, 0.773, **0.795**, 0.768, 0.758, 0.758 for 1.0 to
7.0), so no test evaluations were added and the `guide30` results of round 6 stand. Unpaired, 4.0 to 7.0 against 3.0 is
p 0.39 / 0.24 / 0.24 (Fisher). The scales above 3.0 stay above no guidance (0.710; same episodes: gained 48, lost 25,
p 0.010 at 4.0; 50 / 31, p 0.045 at 5.0; 55 / 36, p 0.059 at 7.0). Most of the drop above 3.0 is on see-through
episodes (0.660 → 0.58, 0.59, 0.57; normal 0.840 → 0.830, 0.813, 0.820).

### Control over three seeds (1200 episodes)

200 episodes per test set and seed. Last column: same checkpoints, same episodes, against the run as trained (sign test).

| run | guidance | seed 42 / 43 / 44 (of 400) | all | normal (930) | see-through (270) | crossover (126) | against the run as trained |
|---|---|---|---|---|---|---|---|
| rgbd (two seeds, 800) | | 261 / 251 / – | 0.640 | 0.798 | 0.094 | 0.774 | |
| rgbdt (two seeds, 800) | | 239 / 262 / – | 0.626 | 0.747 | 0.211 | 0.548 | |
| **nogate_rgbonly** | – | 273 / 245 / 270 | 788 = **0.657** (0.63–0.68) | 0.695 | 0.526 | 0.500 | |
| nogate_rgbonly | 1.5 | 272 / 250 / 288 | 810 = 0.675 | 0.716 | 0.533 | 0.500 | gained 84, lost 62 (p 0.082) |
| nogate_rgbonly | 3.0 | 273 / 258 / 287 | 818 = 0.682 | 0.729 | 0.519 | 0.484 | gained 129, lost 99 (p 0.054) |
| vis_route | – | 282 / 273 / 282 | 837 = 0.698 (0.67–0.72) | 0.741 | 0.548 | 0.643 | |
| vis_route | 1.5 | 304 / 290 / 295 | 889 = 0.741 | 0.795 | 0.556 | 0.738 | gained 93, lost 41 (p 8e-6) |
| vis_route | 3.0 | 311 / 305 / 320 | 936 = 0.780 | 0.840 | 0.574 | 0.778 | gained 160, lost 61 (p 2e-11) |

- **(a) As trained, control against `vis_route`**: 0.657 vs 0.698, Fisher p 0.036; matched by training seed and test
  episode, `vis_route` gained 126 and lost 77 (sign test p 7e-4). Normal 0.695 vs 0.741 (p 0.030), crossover 0.500 vs
  0.643 (p 0.030), see-through 0.526 vs 0.548 (p 0.67). At the seed level (three values each) the difference is not
  significant: 0.657 (sd 0.038) vs 0.698 (sd 0.013), Welch p 0.20, paired by seed p 0.11; the control varies more
  between seeds (245–273 of 400) than `vis_route` (273–282).
- **(b) Gain from guidance in the control**: +1.8 points at 1.5 (p 0.082) and +2.5 at 3.0 (p 0.054); per seed −1 / +5 /
  +18 and 0 / +13 / +17 episodes, so the zero gain of seed 42 in round 6 was the lowest of three. The gain is on normal
  episodes (3.0: gained 100, lost 68, sign test p 0.017; +3.4 points, bootstrap over episodes −0.0 to +6.9), none on
  see-through (gained 29, lost 31) or crossover episodes (12 / 14).
- **(c) Difference of the gains, `vis_route` minus control** (bootstrap over the 400 test episodes, seeds averaged):
  +2.5 points at 1.5 (95% interval −0.1 to +5.2; per seed +5.8, +3.0, −1.2) and **+5.8 points at 3.0** (+2.3 to +9.1;
  per seed +7.2, +4.8, +5.2). At 3.0 by group: normal +6.5 (+2.7 to +10.1), crossover +15.1 (+0.0 to +28.6),
  see-through +3.3 (−4.4 to +11.1). Seed level at 3.0: gains of 29 / 32 / 38 episodes against 0 / 13 / 17, Welch p 0.028.
- **With guidance, control against `vis_route`**: 0.675 vs 0.741 at 1.5 (Fisher p 5e-4); 0.682 vs 0.780 at 3.0
  (p 7e-8; normal 0.729 vs 0.840, p 8e-9; crossover 0.484 vs 0.778, p 2e-6; see-through 0.519 vs 0.574, p 0.23; seed
  level Welch p 0.025). The guided control is level with `vis_route` as trained (0.682 vs 0.698, p 0.43) except on
  crossover episodes (0.484 vs 0.643, p 0.016).
- **Control against the baselines** (their two seeds, 800 episodes): rgbdt 0.626 (p 0.17), rgbd 0.640 (p 0.44). Against
  rgbdt: see-through 0.526 vs 0.211 (p 1e-11), normal 0.695 vs 0.747 (p 0.029), crossover 0.500 vs 0.548 (p 0.57);
  against rgbd: normal 0.695 vs 0.798 (p 6e-6), crossover 0.500 vs 0.774 (p 9e-5). Per seed against the same-seed rgbdt:
  273 vs 239 (p 0.015) at seed 42, 245 vs 262 (p 0.24) at seed 43. With `guide30` the control is 0.682 (rgbdt p 0.011,
  rgbd p 0.059).
- Weights of the control 1 / 1 / 1 in every evaluated step, all three seeds.
- Per test set: control (a) 0.682, (g) 0.632; with `guide30` 0.715 / 0.648 (`vis_route` 0.732 / 0.663 and 0.807 / 0.753).
- Failure outcomes over 1200 episodes: control as trained missed grasp 193, dropped 132, wrong height 55, other 25;
  with `guide30` 167 / 114 / 63 / 32 (`vis_route`: 164 / 111 / 69 / 18 and 101 / 91 / 46 / 20).
- 50 episodes per test set: seed 43 (a) 0.64, (g) 0.58; seed 44 0.62 / 0.62; three seeds 185/300 = 0.617 (normal 0.645,
  see-through 0.515, crossover 0.500; `vis_route` 208/300 = 0.693).

### Why guidance gains more with the routing: offline probe (no training)

`policy/analyze_guidance.py` on the seed-42 checkpoints of `vis_route` and of the control, on expert sequences of the
demo set (two clean frames and the expert's action chunk): the 25 demo episodes that were held out of training (the
validation split of the dataset; 1390 sequences, 640 of them for the action level; 16 episodes with neither condition,
8 see-through, 2 crossover, one of them both) and, for comparison, 3000 sequences of the training episodes (1024 for
the action level).

Measures, fixed before the probe was run:

1. **Noise level.** `e_r`: the denoiser output with the policy's own sensor rule (the control: all sensors on), `e_0`:
   with depth and thermal off, `d = e_r − e_0`, the direction along which guidance extrapolates. The least-squares
   scale `s* = Σ⟨d, ε − e_0⟩ / Σ|d|²` (diffusion steps 5, 20, 40, 60, 80, 95; sequences with a sensor on) says how far
   along `d` the true noise lies: above 1, guidance would lower the denoising error.
2. **Action level.** The full 100-step sampler from the same initial noise and sampler noise for every scale; squared
   error of the 8 executed action steps against the expert's (normalized units), relative to scale 1.

Hypothesis: `s*` above 1 and a lower action error with guidance for `vis_route`, not for the control.

Held-out episodes (95% intervals: bootstrap over episodes):

| group (episodes) | policy | `s*` | action error relative to scale 1: rgb only | 1.5 | 3.0 | 5.0 | 7.0 |
|---|---|---|---|---|---|---|---|
| all (25) | vis_route | 0.86 (0.63–1.07) | 1.56 | 1.01 | 1.07 (0.99–1.20) | 1.10 | 1.17 |
| | control | 0.74 (0.50–1.00) | 1.44 | 1.03 | 1.07 (0.99–1.16) | 1.09 | 1.16 |
| neither (16) | vis_route | 1.01 (0.80–1.22) | 1.85 | 0.98 | 1.03 (0.92–1.20) | 1.06 | 1.09 |
| | control | 0.98 (0.72–1.27) | 1.98 | 0.99 | 0.99 (0.91–1.10) | 1.00 | 1.02 |
| see-through (8) | vis_route | 0.41 (0.15–0.70) | 1.10 | 1.06 | 1.09 (1.02–1.33) | 1.18 | 1.34 |
| | control | 0.38 (0.15–0.67) | 0.82 | 1.10 | 1.21 (1.09–1.33) | 1.26 | 1.42 |
| crossover (2) | vis_route | 0.81 (one episode with a sensor on) | 0.91 | 1.02 | 1.11 | 1.04 | 1.04 |
| | control | 0.03 (−0.03–0.13) | 0.47 | 1.01 | 1.16 (1.04–1.23) | 1.21 | 1.30 |

Training episodes: `s*` 0.99 (0.97–1.01) for `vis_route` and 1.00 (0.98–1.01) for the control, 0.98–1.11 in every
group; the action error rises with the scale in both (×1.30 and ×1.24 at 3.0, ×2.1 and ×2.0 at 7.0).

- **The hypothesis is not supported.** In neither policy does going further along `d` lower the denoising error or the
  open-loop action error. Where both sensors show the glass ("neither"), scale 1 is the least-squares optimum in both
  policies (`s*` 1.01 and 0.98) and the action error is flat up to 3.0 in both (1.03, 0.99). The closed-loop gain of
  `vis_route` is therefore not a better one-shot imitation of the expert's chunk, and this probe does not separate the
  two policies on normal scenes.
- **Where a sensor does not show the glass, the control's sensor direction is of little use on held-out frames**: `s*`
  0.38 (see-through) and 0.03 (crossover), and the rgb-only prediction has a lower action error than the full one
  (×0.82, ×0.47). Guidance scales this direction up (error ×1.21 and ×1.16 at 3.0). In `vis_route` the unseen sensor is
  off and `d` comes from the other one (error ×1.09 and ×1.11 at 3.0). Few held-out episodes (8 and 2).

Exploratory, chosen after seeing the above:

- **Spread across noise draws** (400 held-out sequences, 4 draws each, `*_k4`): guidance narrows the predicted joint targets in
  both policies (standard deviation across draws 0.0076 → 0.0064 at 3.0 for `vis_route`, 0.0076 → 0.0054 for the
  control; the gripper command does not change), so a sharper action distribution does not separate them either.
- **First grasp attempt in the test evaluations** (`closes` in `episodes.json`: horizontal distance to the target and
  height error of the tool at the first gripper close; "first attempt succeeds" = success with a single close). As
  trained → `guide30`, three seeds per policy, matched by seed and episode; "difference" is the change of `vis_route`
  minus the change of the control with a 95% interval (bootstrap over the test episodes).

| group (episodes × seeds) | policy | first close, xy (cm) | first close, height error (cm) | first attempt succeeds | success |
|---|---|---|---|---|---|
| neither (274 × 3) | vis_route | 1.04 → 0.84 | 0.72 → 0.64 | 0.710 → 0.791 | 0.751 → 0.842 |
| | control | 1.27 → 1.04 | 0.81 → 0.77 | 0.681 → 0.704 | 0.718 → 0.751 |
| | difference | +0.03 (−0.06 to +0.15) | −0.04 (−0.09 to +0.03) | +5.7 points (+2.3 to +9.1) | +5.8 (+2.1 to +9.5) |
| crossover (42 × 3) | vis_route | 1.35 → 1.23 | 1.02 → 0.75 | 0.603 → 0.706 | 0.643 → 0.778 |
| | control | 1.91 → 2.44 | 1.37 → 1.52 | 0.468 → 0.444 | 0.500 → 0.484 |
| | difference | −0.65 (−1.18 to −0.16) | −0.42 (−0.66 to −0.18) | +12.7 (−0.8 to +25.4) | +15.1 (+0.8 to +28.6) |
| see-through (90 × 3) | vis_route | 1.99 → 2.48 | 1.12 → 1.08 | 0.515 → 0.507 | 0.548 → 0.574 |
| | control | 2.01 → 2.63 | 1.26 → 1.31 | 0.485 → 0.474 | 0.526 → 0.519 |
| | difference | −0.13 (−0.43 to +0.23) | −0.09 (−0.22 to +0.04) | +0.4 (−6.7 to +7.0) | +3.3 (−4.4 to +11.1) |

  - Crossover: without the gate, guidance moves the first grasp away from the target (1.91 → 2.44 cm; with the gate
    1.35 → 1.23), at every seed (difference −0.55, −0.83, −0.57 cm). This matches the probe: the thermal direction of
    the control is useless on crossover frames and guidance scales it up.
  - See-through: the first grasp moves away in both policies (+0.5 and +0.6 cm) and neither gains.
  - Neither condition, where both policies have all sensors on: the grasp position improves equally (xy −0.20 and
    −0.23 cm), yet the first attempt succeeds more often only with `vis_route` (+8.0 against +2.3 points; difference at
    each seed +5.8, +5.1, +6.2). The larger gain of `vis_route` on normal scenes is reproducible over seeds and is not
    explained by any measure here.

### Control without the rgb-only samples (`nogate_allon`, seed 42, 400 episodes)

| run | all | normal (310) | see-through (90) | crossover (42) | against `nogate_allon` (Fisher) |
|---|---|---|---|---|---|
| **nogate_allon** | 244 = **0.610** (0.56–0.66) | 0.632 | 0.533 | 0.452 | |
| nogate_rgbonly (seed 42) | 273 = 0.682 | 0.710 | 0.589 | 0.500 | p 0.038 |
| vis_route (seed 42) | 282 = 0.705 | 0.739 | 0.589 | 0.690 | p 0.006 |
| static_nohole | 229 = 0.573 | 0.577 | 0.556 | 0.429 | p 0.31 |
| gate_nohole | 250 = 0.625 | 0.665 | 0.489 | 0.452 | p 0.72 |
| rgbdt | 239 = 0.598 | 0.706 | 0.222 | 0.571 | p 0.77 |
| rgbd | 261 = 0.652 | 0.816 | 0.089 | 0.857 | p 0.24 |

- Against the control with the rgb-only samples at the same seed: 0.610 vs 0.682; on the same episodes the control
  gained 52 and lost 23 (sign test p 0.001); normal 0.632 vs 0.710 (p 0.049), see-through and crossover not different
  (p 0.55, 0.83). The control's own seeds range from 0.613 to 0.682, so one seed does not settle this difference.
- Not distinguishable from `static_nohole` (p 0.31; same episodes 59 / 44, p 0.17), `gate_nohole` or rgbdt overall.
  See-through stays at 0.533 against 0.222 for rgbdt: that gain needs neither the routing nor the rgb-only samples.
- Weights 1 / 1 / 1 in every evaluated step and 1.0000 in every training log line. Failures: missed grasp 77, dropped
  43, wrong height 27, other 9.
- 50 episodes per test set: (a) 0.56, (g) 0.56 (normal 0.577, see-through 0.500, crossover 0.250).

### Related work checked (11/17)

Search-engine check of abstracts; TAG, FDP and Evidence-Gated Regularization were read in more detail.

- **Guidance on observations.** TAG (arXiv 2603.24584) guides a vision-language-action policy with the difference
  between its prediction on the observation and on a target-erased one (trained with 10% erased observations; best
  scale 1.25, large scales fail); RGB only, no sensor selection. FDP (arXiv 2509.16830) trains a base policy on a
  prioritized modality and a residual on the others, with a priority fixed for the whole task, and has as a baseline
  one model with the weaker modality dropped with probability 0.2 plus classifier-free guidance, which is the setting
  of the control here. PoCo (RSS 2024) composes sensor-specific policies in the same way. Pearce et al. (ICLR 2023)
  report that classifier-free guidance on observations harms behaviour cloning.
- **Per-frame sensor evidence.** Evidence-Gated Regularization (arXiv 2609.03142) measures per frame how much of the
  target each camera sees and uses it to switch two consistency losses, in training only (RGB and tactile; no sensor
  is switched off at inference, no guidance). Selective Perception (arXiv 2602.15543) learns a router over camera,
  thermal and tactile inputs from VLM labels. "Learning when to see and when to feel" (arXiv 2604.01414) gates
  force / torque by contact and mixes modality-specific noise predictions. MS-Bot (arXiv 2408.01366) sets modality
  priority by task stage.
- **Thermal.** Safe-Night VLA (arXiv 2603.05754) feeds RGB, thermal and depth to a vision-language-action model;
  RGB-thermal glass segmentation (arXiv 2204.05453) and TRansPose (arXiv 2307.05016) are perception work. No
  manipulation policy for transparent objects on RGB, depth and thermal was found.
- **Not found**: the same target-visibility gate in training and at inference combined with guidance toward the
  sensors it keeps on, and a control showing that the gain from sensor guidance depends on that gate. Guidance on
  observations by itself is not new (TAG, FDP's baseline).

### Conclusions of round 7

20. **The guidance scale 3.0 is an interior optimum on the validation episodes** (0.795; 0.768, 0.758, 0.758 at 4, 5
    and 7; 5 and 7 are below 3.0 on the same episodes, p 0.017 and 0.040). The round 6 results with `guide30` stand;
    the caveat "selected at the edge of its grid" is removed.
21. **Without guidance the routing adds about 4 points over its control** (revises conclusion 17): 0.698 against 0.657
    over three seeds each (Fisher p 0.036), on crossover (0.643 vs 0.500) and normal episodes (0.741 vs 0.695), not on
    see-through ones (0.548 vs 0.526). The evidence is weak at the seed level (Welch p 0.20): the control's seeds
    spread over 7 points, and seed 42, the only one in round 6, was its best.
22. **Guidance gains about three times more with the routing; it is not without effect in the control** (revises
    conclusion 16): +8.3 points for `vis_route` and +2.5 for the control at 3.0 (the control's gain: p 0.054, on normal
    episodes only), difference +5.8 points (+2.3 to +9.1), +7.2 / +4.8 / +5.2 at the three seeds. At 1.5 the difference
    (+2.5 points) no longer excludes 0. With guidance the gap is 0.780 against 0.682 (p 7e-8; seed level p 0.025).
23. **The control is not above rgbdt over seeds** (revises conclusion 18): 0.657 against 0.626 (p 0.17) and against
    rgbd 0.640 (p 0.44). Its profile is the one of round 6: see-through far above both baselines (0.526 vs 0.211 and
    0.094), normal below both (0.695 vs 0.747 and 0.798).
24. **Why guidance gains more with the routing is answered in part.** The hypothesis that extrapolating along the
    sensor direction lowers the imitation error for `vis_route` only is not supported: on held-out expert sequences
    scale 1 is the optimum for both policies. What the probe and the test episodes do show: where a sensor does not
    show the glass, the control keeps that sensor on, its contribution there is useless on held-out frames, and
    guidance scales it up, so that on crossover episodes the first grasp moves away from the target (1.91 → 2.44 cm,
    all three seeds) while with the routing it does not (1.35 → 1.23 cm). On scenes where both sensors show the glass
    the larger gain of `vis_route` (first attempt succeeds +8.0 against +2.3 points, all three seeds) is not explained.
25. **The 10% rgb-only samples account for the control's margin over `static_nohole` at seed 42** (0.682 → 0.610
    without them, p 0.038; `static_nohole` 0.573, p 0.31 against `nogate_allon`), on normal episodes. One seed, and
    the difference is the size of the control's seed-to-seed range. The see-through gain over rgbdt (0.53 vs 0.22)
    remains without the rgb-only samples and without the routing.

Caveats: rgbd and rgbdt have two seeds; the guidance scale was selected for `vis_route` and applied to the control
unchanged; seed-level tests have three values per side; the first-grasp and noise-draw analyses were chosen after
seeing the results; the probe uses the seed-42 checkpoints and 25 held-out demo episodes (8 see-through, 2 crossover);
`nogate_allon` has one seed; p values are not corrected for the number of comparisons.

### Next steps (proposals; 1, 2 and 4 were approved on 11/17 and are run as round 8)

1. Third seed of rgbd and rgbdt (2 trainings, 2 evaluations), so that every comparison has three seeds.
2. Robustness of the gate decision (evaluations only): thresholds moved, masks degraded, flags flipped with a given
   probability, for `vis_route` as trained and with `guide30`.
3. A control with single-sensor dropout at the rates of the routing but independent of visibility (depth off in about
   25% of the training samples, thermal in about 7%, plus the 10% rgb-only samples), then guidance: separates "trained
   with single sensors off" from "off where the glass is not visible", the open question on normal scenes.
4. An external baseline: composition of sensor-specific policies (PoCo / Modality-Composable Diffusion Policy style).
5. The guidance scale of the control on the validation episodes (3.0 was chosen for `vis_route`).
6. Partially visible targets and the third seed of learned attention weighting, as proposed above.

## Round 8: third baseline seed, gate errors at inference, composition baseline (11/17 13:59 – 23:19 KST)

### Runs

Two trainings and 31 evaluations; all 33 queued jobs finished at the first attempt.

- **Baselines, third seed**: `L0_bench_rgbd_500demos_seed44` and `L0_bench_rgbdt_500demos_seed44`, plain Diffusion
  Policy with the benchmark command of seeds 42 and 43 and `training.seed=44`
  (`GROUP=fusion_bench TAG=seed44 EXTRA="training.seed=44" bash scripts/run_experiment.sh L0_bench <rgbd|rgbdt> 500`),
  epoch 100 without interruption. Their results are here, not in `benchmark/`; 200 episodes per test set in `n200/`.
- **Gate errors**, `n200/L0_bench_rgbdt_500demos_vis_route__<tag>[_g30]/`: the `vis_route` checkpoint of seed 42 on
  the test episodes with wrong visibility flags, as trained and with `--guidance 3.0` (19 evaluations; the row "both
  always on" as trained is `__all_on` of rounds 3–5). Nothing is retrained.
- **Composition**, `val100/compose_w{25,50,75}[_seed43]/`, `n200/compose_w75[_seed43]/`, `n200/compose_poe[_seed43]/`:
  the plain rgbd, rgbt and rgb checkpoints of the benchmark (seeds 42 and 43) combined at inference (10 evaluations).

Code added, all opt-in (`python -m policy.test_gating` passes; evaluations without the new options are unchanged
apart from three `null` entries in `eval_log.json`):

- `policy/eval.py`: `--gate_error flip=P` inverts each of the two visibility flags with probability P, independently
  per decision step, environment and sensor; `--gate_error depth=0|1` / `thermal=0|1` forces a flag in every step;
  `--vis_thr D,T` recomputes the flags with other thresholds than the 15 / 30 of training; `--mask_noise shift=K`
  moves the predicted glass mask of each frame by an offset drawn uniformly from [−K, K]² pixels, `--mask_noise drop=P`
  empties it with probability P (a frame without a mask counts as visible in both sensors). The flags are the only
  path from the gate to the policy (the cue is not in the conditioning), so these cover every way the gate can be
  wrong. Implemented in `policy/gated_policy.py` (`gate_error`) and `policy/obs_util.py` (`degrade_mask`).
- `policy/compose_policy.py`, `policy/eval_compose.py`: composition of separately trained policies at the
  noise-prediction level, eps = Σ w_i eps_i(x_t, t, obs_i), every member with its own encoder and observation keys,
  one shared scheduler step (the distribution-level composition of Modality-Composable Diffusion Policy,
  arXiv 2503.12466, and of the sensor policies of PoCo). With one-hot weights it reproduces a member exactly.

Fixed before the results:

- Baselines: all comparisons of round 7 are recomputed with three seeds on each side.
- Gate errors: the eleven conditions of the table below, each as trained and with `guide30`, on seed 42 only; the
  realized flag errors are reported with the success rates.
- Composition: the weighted mean of rgbd and rgbt with rgbd weight 0.25, 0.5 or 0.75, selected on the validation
  episodes of round 6 (highest success pooled over both seeds and both test sets, ties to 0.5), then evaluated on the
  test episodes; the product rgbd × rgbt / rgb (weights 1, 1, −1) evaluated on the test episodes directly; seeds 42
  and 43 (rgbt and rgb have no third seed).

### Baselines over three seeds (1200 episodes)

200 episodes per test set and seed. The `vis_route` and control rows are those of rounds 6 and 7.

| run | guidance | seed 42 / 43 / 44 (of 400) | all | normal (930) | see-through (270) | crossover (126) | neither (822) |
|---|---|---|---|---|---|---|---|
| **rgbd** | | 261 / 251 / 237 | 749 = **0.624** (0.60–0.65) | 0.782 | 0.081 | 0.754 | 0.773 |
| **rgbdt** | | 239 / 262 / 235 | 736 = **0.613** (0.59–0.64) | 0.731 | 0.207 | 0.476 | 0.757 |
| nogate_rgbonly | – | 273 / 245 / 270 | 788 = 0.657 | 0.695 | 0.526 | 0.500 | 0.718 |
| nogate_rgbonly | 3.0 | 273 / 258 / 287 | 818 = 0.682 | 0.729 | 0.519 | 0.484 | 0.751 |
| vis_route | – | 282 / 273 / 282 | 837 = 0.698 (0.67–0.72) | 0.741 | 0.548 | 0.643 | 0.751 |
| vis_route | 3.0 | 311 / 305 / 320 | 936 = 0.780 (0.76–0.80) | 0.840 | 0.574 | 0.778 | 0.842 |

"Matched": the runs of the same training seed number on the same test episodes (sign test). Seed level: the three
success counts per side, Welch t-test / t-test paired by seed number.

| comparison | all (Fisher) | matched | seed level | normal | see-through | crossover |
|---|---|---|---|---|---|---|
| vis_route vs rgbd | 0.698 vs 0.624, p 2e-4 | gained 251, lost 163 (p 2e-5) | 0.036 / 0.065 | 0.741 vs 0.782 (p 0.044) | 0.548 vs 0.081 (p 1e-33) | 0.643 vs 0.754 (p 0.074) |
| vis_route vs rgbdt | 0.698 vs 0.613, p 2e-5 | gained 249, lost 148 (p 5e-7) | 0.045 / 0.098 | 0.741 vs 0.731 (p 0.67) | 0.548 vs 0.207 (p 3e-16) | 0.643 vs 0.476 (p 0.011) |
| vis_route, 3.0 vs rgbd | 0.780 vs 0.624, p 8e-17 | gained 282, lost 95 (p 1e-22) | 0.003 / 0.027 | 0.840 vs 0.782 (p 0.002) | 0.574 vs 0.081 (p 1e-36) | 0.778 vs 0.754 (p 0.77) |
| vis_route, 3.0 vs rgbdt | 0.780 vs 0.613, p 7e-19 | gained 287, lost 87 (p 5e-26) | 0.006 / 0.033 | 0.840 vs 0.731 (p 1e-8) | 0.574 vs 0.207 (p 2e-18) | 0.778 vs 0.476 (p 1e-6) |
| control vs rgbd | 0.657 vs 0.624, p 0.11 | gained 246, lost 207 (p 0.074) | 0.32 / 0.37 | 0.695 vs 0.782 (p 2e-5) | 0.526 vs 0.081 (p 4e-31) | 0.500 vs 0.754 (p 5e-5) |
| control vs rgbdt | 0.657 vs 0.613, p 0.031 | gained 235, lost 183 (p 0.013) | 0.23 / 0.42 | 0.695 vs 0.731 (p 0.091) | 0.526 vs 0.207 (p 2e-14) | 0.500 vs 0.476 (p 0.80) |
| control, 3.0 vs rgbd | 0.682 vs 0.624, p 0.004 | gained 254, lost 185 (p 0.001) | 0.11 / 0.23 | 0.729 vs 0.782 (p 0.010) | 0.519 vs 0.081 (p 3e-30) | 0.484 vs 0.754 (p 2e-5) |
| control, 3.0 vs rgbdt | 0.682 vs 0.613, p 5e-4 | gained 239, lost 157 (p 4e-5) | 0.083 / 0.24 | 0.729 vs 0.731 (p 0.96) | 0.519 vs 0.207 (p 6e-14) | 0.484 vs 0.476 (p 1) |
| rgbdt vs rgbd | 0.613 vs 0.624, p 0.61 | gained 149, lost 162 (p 0.50) | | 0.731 vs 0.782 (p 0.013) | 0.207 vs 0.081 (p 4e-5) | 0.476 vs 0.754 (p 9e-6) |

- Seed 44 is the lowest seed of both baselines (rgbd 0.593, rgbdt 0.588), so their pooled values moved down from the
  two-seed ones (rgbd 0.640 → 0.624, rgbdt 0.626 → 0.613). Standard deviation over seeds: rgbd 0.030, rgbdt 0.036,
  control 0.038, `vis_route` 0.013 as trained and 0.019 with `guide30`.
- `vis_route` as trained is above rgbd through the see-through episodes only; on normal episodes it is below rgbd
  (0.741 vs 0.782, p 0.044). With `guide30` it is above rgbd on normal episodes too (0.840, p 0.002) and level on
  crossover ones.
- rgbdt against rgbd over three seeds: no difference overall; rgbdt is better on see-through episodes and worse on
  normal and crossover ones (0.476 vs 0.754; 14/42 at seed 44).
- Per test set: rgbd (a) 0.642, (g) 0.607; rgbdt 0.638 / 0.588 (`vis_route` 0.732 / 0.663, with `guide30` 0.807 / 0.753).
- Failure outcomes over 1200 episodes: rgbd missed grasp 252, other 99, dropped 74, wrong height 25, wrong object 1;
  rgbdt 221 / 82 / 100 / 58 / 3.
- 50 episodes per test set, seed 44: rgbd (a) 0.62, (g) 0.44; rgbdt 0.70 / 0.60.

### Gate errors at inference (`vis_route`, seed 42, 400 episodes per cell)

Realized flag errors, as shares of the decision steps of the runs as trained (the guided runs are within 2 points of
these). "Truth" is the episode label: depth should be off on see-through episodes, thermal on crossover episodes.

| error | depth on, see-through | depth off, normal | thermal on, crossover | thermal off, otherwise |
|---|---|---|---|---|
| none | 0.1% | 0.6% | 1.0% | 0.6% |
| flags flipped, 10% | 11.2% | 10.9% | 11.0% | 11.0% |
| flags flipped, 30% | 31.0% | 29.3% | 31.1% | 29.6% |
| depth always off | 0% | 100% | 3.5% | 1.0% |
| thermal always off | 0.4% | 0.8% | 0% | 100% |
| both always on | 100% | 0% | 100% | 0% |
| thresholds × 0.5 (7.5 / 15) | 4.0% | 0.3% | 5.3% | 0.3% |
| thresholds × 2 (30 / 60) | 0.0% | 1.6% | 1.0% | 8.6% |
| mask shifted up to 1 px | 0.9% | 2.5% | 3.5% | 3.1% |
| mask shifted up to 2 px | 1.1% | 38.5% | 5.8% | 21.7% |
| mask missing in 30% of the frames | 31.5% | 0.6% | 30.8% | 0.7% |

Success, with the same test episodes in every cell. "vs none": episodes lost / gained against the row "none" of the
same guidance (sign test). The baselines at this seed on the same episodes are rgbdt 0.598 and rgbd 0.652.

| error | as trained | normal (310) | see-through (90) | crossover (42) | vs none | guide 3.0 | normal | see-through | crossover | vs none |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 282 = 0.705 | 0.739 | 0.589 | 0.690 | | 311 = 0.777 | 0.829 | 0.600 | 0.786 | |
| flags flipped, 10% | 272 = 0.680 | 0.729 | 0.511 | 0.643 | 26 / 16 (p 0.16) | 301 = 0.752 | 0.826 | 0.500 | 0.667 | 31 / 21 (p 0.21) |
| flags flipped, 30% | 244 = 0.610 | 0.684 | 0.356 | 0.524 | 54 / 16 (p 6e-6) | 272 = 0.680 | 0.803 | 0.256 | 0.548 | 59 / 20 (p 1e-5) |
| depth always off | 230 = 0.575 | 0.568 | 0.600 | 0.500 | 62 / 10 (p 3e-10) | 257 = 0.642 | 0.658 | 0.589 | 0.500 | 69 / 15 (p 2e-9) |
| thermal always off | 276 = 0.690 | 0.723 | 0.578 | 0.690 | 28 / 22 (p 0.48) | 306 = 0.765 | 0.835 | 0.522 | 0.762 | 39 / 34 (p 0.64) |
| both always on | 234 = 0.585 | 0.703 | 0.178 | 0.381 | 63 / 15 (p 4e-8) | 252 = 0.630 | 0.784 | 0.100 | 0.262 | 68 / 9 (p 2e-12) |
| thresholds × 0.5 | 289 = 0.723 | 0.758 | 0.600 | 0.762 | 2 / 9 (p 0.065) | 314 = 0.785 | 0.845 | 0.578 | 0.857 | 11 / 14 (p 0.69) |
| thresholds × 2 | 282 = 0.705 | 0.742 | 0.578 | 0.714 | 7 / 7 (p 1) | 312 = 0.780 | 0.839 | 0.578 | 0.833 | 23 / 24 (p 1) |
| mask shifted up to 1 px | 282 = 0.705 | 0.739 | 0.589 | 0.714 | 5 / 5 (p 1) | 316 = 0.790 | 0.855 | 0.567 | 0.857 | 15 / 20 (p 0.50) |
| mask shifted up to 2 px | 266 = 0.665 | 0.694 | 0.567 | 0.690 | 31 / 15 (p 0.026) | 304 = 0.760 | 0.787 | 0.667 | 0.690 | 30 / 23 (p 0.41) |
| mask missing, 30% | 261 = 0.652 | 0.726 | 0.400 | 0.595 | 39 / 18 (p 0.008) | 285 = 0.713 | 0.819 | 0.344 | 0.524 | 43 / 17 (p 0.001) |

- **Small errors cost nothing measurable**: thresholds halved or doubled and a mask off by one pixel change at most 14
  of 400 episodes as trained (net 0 to +7) and leave the guided result at 0.780–0.790.
- **Random flips**: 10% costs 2.5 points in both settings (not significant); 30% costs 9.5 and 9.7 points and brings
  the policy as trained to the level of the baselines (0.610; rgbdt 0.598, p 0.77), the guided one to 0.680.
- **The costly error is a sensor left on where it does not show the glass.** On see-through episodes: 0.589 as
  trained → 0.400 with depth on in 31% of the steps (mask missing), 0.356 (flips, 30%), 0.178 when always on. With
  guidance the same errors cost more (0.600 → 0.344, 0.256, 0.100), and on crossover episodes with thermal always on
  0.786 → 0.262: guidance scales up the contribution of a sensor that is wrongly on. Normal episodes change little
  under the two errors that only switch sensors on (mask missing 0.726, always on 0.703, against 0.739 as trained;
  0.819 and 0.784 against 0.829 guided).
- **A sensor wrongly off costs less per step**: depth off in 38% of the normal steps and thermal in 22% (mask shifted
  up to 2 px, half the width of the 4 px mask) lowers normal episodes from 0.739 to 0.694 (0.829 → 0.787 guided);
  depth off in all steps lowers them to 0.568 (0.658 guided).
- **Thermal always off changes nothing measurable**: 0.690 against 0.705 (28 lost, 22 gained, p 0.48), guided 0.765
  against 0.777 (39 / 34, p 0.64). On the episodes where both sensors show the glass ("neither", 274) it is 0.726
  against 0.745 as trained and 0.843 against 0.832 guided; on see-through episodes, where the policy then runs on
  rgb alone, 0.578 against 0.589 (guided 0.522 against 0.600; 16 lost, 9 gained). Depth always off costs 13 points.
- **The gain from guidance is kept under every error**: +6.0 to +9.5 points (sign test p < 0.01 each), +4.5 with both
  sensors always on (p 0.033).
- Against the baselines at seed 42 (Fisher): as trained the policy is above rgbdt for no error, flips at 10%, thermal
  always off, both threshold changes and the 1 px shift (p ≤ 0.018), and no longer for the other five (p 0.057–0.77);
  it is not significantly below rgbdt in any condition (lowest: depth always off 0.575, both always on 0.585), and
  below rgbd only with depth always off (0.575 vs 0.652, p 0.029). Guided, it is above
  rgbdt in every condition but depth always off and both always on (p 0.22, 0.38), and above rgbd (p ≤ 0.003) except
  for flips at 30%, depth always off, both always on and mask missing (p 0.45, 0.82, 0.56, 0.081).

### Composition of sensor-specific policies (seeds 42 and 43)

Members: the plain rgbd, rgbt and rgb policies of the benchmark, same demos and action normalizer. Validation, 100
episodes per test set and seed (seeds 200000–200099; `vis_route` as trained reaches 284/400 = 0.710 on them):

| rgbd / rgbt weight | seed 42 (a) / (g) | seed 43 (a) / (g) | pooled (400) |
|---|---|---|---|
| 0.25 / 0.75 | 65 / 62 | 48 / 55 | 230 = 0.575 |
| 0.5 / 0.5 | 69 / 70 | 58 / 59 | 256 = 0.640 |
| 0.75 / 0.25 | 68 / 66 | 63 / 62 | **259 = 0.647** |

**Selected: 0.75 / 0.25** by the fixed rule (three episodes above 0.5 / 0.5, which was not evaluated on the test
episodes). Test, 200 episodes per test set and seed (800 per row; the other rows are the same two seeds):

| policy | seed 42 / 43 (of 400) | all | normal (620) | see-through (180) | crossover (84) | neither (548) |
|---|---|---|---|---|---|---|
| rgb | 161 / 150 | 0.389 | 0.395 | 0.367 | 0.417 | 0.387 |
| rgbt | 228 / 177 | 0.506 | 0.508 | 0.500 | 0.262 | 0.540 |
| rgbd | 261 / 251 | 0.640 | 0.798 | 0.094 | 0.774 | 0.788 |
| rgbdt | 239 / 262 | 0.626 | 0.747 | 0.211 | 0.548 | 0.763 |
| **mean of rgbd and rgbt, 0.75 / 0.25** | 271 / 246 | 517 = **0.646** (0.61–0.68) | 0.779 | 0.189 | 0.667 | 0.781 |
| **product rgbd × rgbt / rgb** | 240 / 179 | 419 = **0.524** (0.49–0.56) | 0.634 | 0.144 | 0.286 | 0.673 |
| nogate_rgbonly | 273 / 245 | 0.647 | 0.681 | 0.533 | 0.488 | 0.703 |
| vis_route | 282 / 273 | 0.694 | 0.734 | 0.556 | 0.655 | 0.743 |
| vis_route, guide 3.0 | 311 / 305 | 0.770 | 0.827 | 0.572 | 0.762 | 0.832 |

- **The weighted mean is at the level of its better member and of the joint rgbdt policy**: 0.646 against rgbd 0.640
  (p 0.84; same episodes 53 gained, 48 lost) and rgbdt 0.626 (p 0.44); above rgbt (p 2e-8). On see-through episodes
  it is 0.189, between rgbd (0.094, p 0.015) and rgbt (0.500, p 6e-10): a fixed weight cannot hand the scene to the
  sensor that shows the glass.
- **`vis_route` against the weighted mean**: 0.694 vs 0.646 as trained (Fisher p 0.049; same seeds and episodes
  142 gained, 104 lost, p 0.018), through the see-through episodes (0.556 vs 0.189, p 6e-13); on normal episodes the
  mean is ahead (0.779 vs 0.734, p 0.074). With `guide30`: 0.770 vs 0.646 (p 7e-8; 163 / 64, p 4e-11), normal 0.827
  vs 0.779 (p 0.038), crossover 0.762 vs 0.667 (p 0.23).
- **The product is below the better of its two sensor members**: 0.524, below rgbd (p 3e-6) and rgbdt (p 4e-5),
  level with rgbt (0.506, p 0.52); crossover 0.286, see-through 0.144. Its two seeds differ by 61 episodes, as do
  those of rgbt (51).
- Per test set: mean (a) 0.675, (g) 0.618; product 0.535 / 0.512. Failure outcomes over 800 episodes: mean missed
  grasp 182, dropped 57, other 27, wrong height 17; product 206 / 64 / 80 / 25, wrong object 6.

### Conclusions of round 8

26. **With three seeds of the baselines the result stands** (removes the first caveat of round 7): rgbd 0.624 and
    rgbdt 0.613 over 1200 episodes each; `vis_route` 0.698 as trained (p 2e-4 and 2e-5; seed level p 0.036 and 0.045)
    and 0.780 with guidance at 3.0 (p 8e-17 and 7e-19; seed level p 0.003 and 0.006). As trained the margin over rgbd
    comes from the see-through episodes alone, with normal episodes below rgbd (0.741 vs 0.782, p 0.044); with
    guidance `vis_route` is above rgbd on normal episodes as well (0.840).
27. **The control is above rgbdt on pooled episodes, not at the seed level** (revises conclusion 23 in part): 0.657
    against 0.613 (p 0.031; seed level p 0.23) and against rgbd 0.624 (p 0.11).
28. **The gate tolerates small errors**: thresholds halved or doubled, a mask off by one pixel and 10% of the flags
    flipped leave the guided result at 0.75–0.79 (0.777 without errors). With 30% of the flags flipped the policy as
    trained falls to the level of the baselines (0.610) and the guided one to 0.680.
29. **The error that matters is a sensor on where it does not show the glass, and guidance makes it worse there**:
    see-through episodes fall from 0.59 to 0.40 / 0.36 / 0.18 with depth wrongly on in 31% / 31% / 100% of the steps,
    and to 0.34 / 0.26 / 0.10 with guidance. A sensor wrongly off costs less (depth off in 38% of the normal steps:
    −4.5 points on normal episodes). A gate for this policy should therefore lean towards "off" when unsure, and a
    missing mask should not default to "on" as it does here.
30. **Switching the thermal sensor off at inference changes nothing measurable in this checkpoint**: with thermal off
    in every step the result is 0.690 (0.705) as trained and 0.765 (0.777) guided, also on the see-through episodes,
    where the policy then acts on rgb alone (0.578 vs 0.589 as trained). Depth is the sensor the policy needs on normal scenes (always
    off: 0.575). What the comparisons with rgbd and rgbdt show is the value of switching depth off where it sees
    through the glass plus the rgb path of this policy; a benefit of thermal is not shown. One seed.
31. **The gain from guidance does not depend on a correct gate**: +6.0 to +9.5 points under every error condition
    (+4.5 with both sensors always on).
32. **Composing sensor-specific policies does not reach `vis_route`**: the validation-selected mean of rgbd and rgbt
    is 0.646, level with rgbd and rgbdt, below `vis_route` as trained (0.694, p 0.049) and with guidance (0.770,
    p 7e-8), and its see-through success stays at 0.19; the product with the rgb prior divided out is 0.524.

Caveats: the gate errors were run on seed 42 only, with one draw of the random errors, and the random errors are
independent between decision steps (real mask errors persist over frames; the "always" rows bound that case); the
composition has two seeds, members that are weak on their own (rgbt 0.506), three convex weights and one product,
and a weight selected by three validation episodes; "matched" pairs runs by seed number only; seed-level tests have
three values per side; p values are not corrected for the number of comparisons.

### Next steps (proposals, not run)

1. Does the policy need thermal at all? Follow-up of conclusion 30: the thermal-off evaluation on seeds 43 and 44
   (4 evaluations), and a `vis_route` trained on rgb + depth only (three seeds, 3 trainings).
2. The gate errors on seeds 43 and 44 (the 19 evaluations, twice).
3. A gate that leans towards "off" (conclusion 29): a missing mask keeps the previous flags instead of switching both
   sensors on; evaluations only.
4. Composition with per-scene weights from the same visibility flags (rgbd where depth shows the glass, rgbt
   otherwise), as a stronger external baseline; evaluations only. A third seed of rgbt and rgb (2 trainings) for it.
5. Open from round 7: the control with visibility-independent single-sensor dropout; the control's own guidance
   scale; partially visible targets; the third seed of learned attention weighting.

## Round 9: thermal input, gate errors over three seeds, missing masks, switched composition (11/18 01:26 – 11/19 00:10)

### Runs

Three trainings and 55 evaluations (proposals 1–4 of round 8). One evaluation (`flip10_g30`, seed 43) was stopped
twice by the memory guard while another account loaded several models and finished at the third attempt; everything
else finished at the first.

- **`vis_route` on rgb + depth**, `L0_bench_rgbd_500demos_vis_route[_seed43|_seed44]`: the `vis_route` command with
  the rgbd input
  (`TRAIN="policy/train_gated.sh gate" TAG=vis_route[_seedNN] EXTRA="<vis_route overrides> [training.seed=NN]"
  bash scripts/run_experiment.sh L0_bench rgbd 500`),
  epoch 100 without interruption; depth is switched off where the glass is not visible in it, 10% rgb-only samples,
  no thermal image in the policy's input. The glass mask of the gate still comes from the rgb + depth + thermal
  segmentation network of rounds 3–5, so this removes thermal from the policy, not from the system. 200 episodes per
  test set in `n200/`, as trained and `__guide30`.
- **Thermal always off** on the `vis_route` checkpoints of seeds 43 and 44 (`__toff`, `__toff_g30`), and the other
  **gate errors** of round 8 on the same two seeds (39 evaluations with the thermal ones), same options and tags.
- **Missing masks keep the last flags** (`--route_hold`), evaluations only: `__mdrop30_hold[_g30]` on the three
  seeds, `__hold[_g30]` (no error) on seed 42.
- **Switched composition**, `n200/compose_switch[_seed43]/`: the plain rgbd and rgbt policies of the benchmark, one of
  them acting per decision (`policy/eval_compose.py -c RGBD -c RGBT --switch <vis_route checkpoint of the seed>`).

Code added, all opt-in (`python -m policy.test_gating` passes; existing configurations give identical outputs):

- `policy/obs_util.py`: `compute_cues` without a thermal image (the thermal entries of the cue are then those of a
  sensor that never shows the glass; the depth entries are unchanged).
- `policy/eval.py --route_hold` (`policy/gated_policy.py`): a frame without a glass mask keeps the flags of the last
  frame of the episode that had one; before the first such frame both sensors are off. Without it such a frame has
  both sensors on (conclusion 29).
- `policy/eval_compose.py --switch CKPT` (`policy/compose_policy.py`): the first member acts where the depth flag is
  on, the second where it is off, per environment and decision, with the visibility cue computed as for the
  `route=visible` checkpoint CKPT (only its cue settings are read); `episodes.json` has the member weights and the
  cue of every decision.

Fixed before the results: the conditions, seeds and tags above; guidance 3.0 as selected in round 6; for the
composition the members and the rule (rgbd where the depth flag is on, rgbt elsewhere), no weight and nothing else
selected, seeds 42 and 43 (rgbt has no third seed).

### Does the policy need the thermal image? (three seeds, 1200 episodes per row)

| run | guidance | seed 42 / 43 / 44 (of 400) | all | normal (930) | see-through (270) | crossover (126) |
|---|---|---|---|---|---|---|
| vis_route | – | 282 / 273 / 282 | 837 = 0.698 | 0.741 | 0.548 | 0.643 |
| vis_route, thermal always off | – | 276 / 268 / 263 | 807 = **0.672** | 0.720 | 0.507 | 0.643 |
| **vis_route trained on rgb + depth** | – | 270 / 262 / 263 | 795 = **0.662** (0.64–0.69) | 0.715 | 0.481 | 0.611 |
| vis_route | 3.0 | 311 / 305 / 320 | 936 = 0.780 | 0.840 | 0.574 | 0.778 |
| vis_route, thermal always off | 3.0 | 306 / 300 / 321 | 927 = **0.772** | 0.855 | 0.489 | 0.786 |
| **vis_route trained on rgb + depth** | 3.0 | 310 / 301 / 300 | 911 = **0.759** (0.73–0.78) | 0.835 | 0.496 | 0.778 |

- **Thermal off at inference, as trained**: 0.698 → 0.672 (same episodes 98 lost, 68 gained, p 0.024; every seed
  lower; seed level Welch p 0.11, paired p 0.16). Seed 42 alone (round 8) showed 0.705 → 0.690, p 0.48.
- **Thermal off at inference, with guidance**: 0.780 → 0.772 overall (110 lost, 101 gained, p 0.58), but the
  see-through episodes fall from 0.574 to 0.489 (51 lost, 28 gained, p 0.013) while the normal ones rise from 0.840
  to 0.855 (59 lost, 73 gained, p 0.26).
- **Trained without the thermal image**: 0.662 as trained, below `vis_route` on the same episodes (144 lost, 102
  gained, p 0.009; Fisher p 0.073; seed level Welch p 0.025, paired p 0.031), on see-through episodes 0.481 against
  0.548 (44 lost, 26 gained, p 0.041). With guidance 0.759 against 0.780 (143 lost, 118 gained, p 0.14; Fisher
  p 0.25; seed level p 0.20 / 0.29), see-through 0.496 against 0.574 (57 lost, 36 gained, p 0.038), normal 0.835
  against 0.840, crossover 0.778 in both.
- Trained without thermal against switching it off at inference: 0.759 against 0.772 with guidance (p 0.47),
  see-through 0.496 against 0.489.
- **Against the baselines the rgb + depth policy keeps most of the margin**: as trained 0.662 against rgbd 0.624
  (Fisher p 0.055; matched 230 gained, 184 lost, p 0.027) and rgbdt 0.613 (p 0.014; 238 / 179, p 0.004), level with
  the control (0.657, p 0.80); with guidance 0.759 against rgbd (p 1e-12; seed level p 0.007), rgbdt (p 2e-14; seed
  level p 0.012) and the guided control 0.682 (p 3e-5; seed level p 0.052 / 0.077).
- **Guidance gains as much without thermal**: +9.7 points for the rgb + depth policy (45 lost, 161 gained), +8.2 for
  `vis_route`, +10.0 for `vis_route` with thermal off, +2.5 for the control.
- 50 episodes per test set at epoch 100, rgb + depth: (a) 0.62 / 0.54 / 0.62, (g) 0.70 / 0.68 / 0.54 for seeds 42 /
  43 / 44.

### Gate errors at inference over three seeds (`vis_route`, 1200 episodes per cell)

The conditions of round 8 on the checkpoints of seeds 42, 43 and 44; the same test episodes in every cell. Realized
flag errors as shares of the decision steps of the runs as trained (the guided runs are within 2 points):

| error | depth on, see-through | depth off, normal | thermal on, crossover | thermal off, otherwise |
|---|---|---|---|---|
| none | 0.5% | 0.7% | 1.5% | 0.7% |
| flags flipped, 10% | 11.2% | 11.0% | 11.4% | 11.0% |
| flags flipped, 30% | 31.0% | 29.3% | 31.9% | 29.7% |
| depth always off | 0% | 100% | 3.3% | 1.0% |
| thermal always off | 0.4% | 0.9% | 0% | 100% |
| both always on | 100% | 0% | 100% | 0% |
| thresholds × 0.5 (7.5 / 15) | 4.5% | 0.5% | 6.4% | 0.3% |
| thresholds × 2 (30 / 60) | 0.2% | 2.1% | 1.0% | 8.7% |
| mask shifted up to 1 px | 0.7% | 2.7% | 3.5% | 3.1% |
| mask shifted up to 2 px | 1.9% | 38.0% | 6.1% | 21.6% |
| mask missing in 30% of the frames | 29.8% | 0.4% | 33.0% | 0.7% |

"vs none": episodes lost / gained against the row "none" of the same guidance, matched by seed and episode (sign
test). Baselines over the same three seeds: rgbd 0.624, rgbdt 0.613.

| error | as trained | seed 42 / 43 / 44 | normal (930) | see-through (270) | crossover (126) | vs none | guide 3.0 | seed 42 / 43 / 44 | normal | see-through | crossover | vs none |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 837 = 0.698 | 282 / 273 / 282 | 0.741 | 0.548 | 0.643 | | 936 = 0.780 | 311 / 305 / 320 | 0.840 | 0.574 | 0.778 | |
| flags flipped, 10% | 803 = 0.669 | 272 / 260 / 271 | 0.718 | 0.500 | 0.611 | 77 / 43 (p 0.002) | 900 = 0.750 | 301 / 291 / 308 | 0.822 | 0.504 | 0.730 | 93 / 57 (p 0.004) |
| flags flipped, 30% | 735 = 0.613 | 244 / 243 / 248 | 0.674 | 0.400 | 0.516 | 148 / 46 (p 1e-13) | 814 = 0.678 | 272 / 262 / 280 | 0.791 | 0.289 | 0.571 | 172 / 50 (p 7e-17) |
| depth always off | 690 = 0.575 | 230 / 225 / 235 | 0.582 | 0.552 | 0.444 | 173 / 26 (p 8e-28) | 740 = 0.617 | 257 / 234 / 249 | 0.635 | 0.552 | 0.437 | 228 / 32 (p 1e-37) |
| thermal always off | 807 = 0.672 | 276 / 268 / 263 | 0.720 | 0.507 | 0.643 | 98 / 68 (p 0.024) | 927 = 0.772 | 306 / 300 / 321 | 0.855 | 0.489 | 0.786 | 110 / 101 (p 0.58) |
| both always on | 723 = 0.603 | 234 / 248 / 241 | 0.720 | 0.196 | 0.444 | 159 / 45 (p 4e-16) | 762 = 0.635 | 252 / 254 / 256 | 0.792 | 0.093 | 0.333 | 198 / 24 (p 3e-35) |
| thresholds × 0.5 | 840 = 0.700 | 289 / 272 / 279 | 0.744 | 0.548 | 0.667 | 17 / 20 (p 0.74) | 938 = 0.782 | 314 / 306 / 318 | 0.846 | 0.559 | 0.817 | 29 / 31 (p 0.90) |
| thresholds × 2 | 836 = 0.697 | 282 / 271 / 283 | 0.739 | 0.552 | 0.643 | 21 / 20 (p 1) | 941 = 0.784 | 312 / 307 / 322 | 0.846 | 0.570 | 0.802 | 44 / 49 (p 0.68) |
| mask shifted up to 1 px | 833 = 0.694 | 282 / 273 / 278 | 0.738 | 0.544 | 0.643 | 12 / 8 (p 0.50) | 941 = 0.784 | 316 / 310 / 315 | 0.845 | 0.574 | 0.786 | 34 / 39 (p 0.64) |
| mask shifted up to 2 px | 767 = 0.639 | 266 / 245 / 256 | 0.666 | 0.548 | 0.571 | 109 / 39 (p 7e-9) | 889 = 0.741 | 304 / 289 / 296 | 0.789 | 0.574 | 0.675 | 102 / 55 (p 2e-4) |
| mask missing, 30% | 790 = 0.658 | 261 / 258 / 271 | 0.735 | 0.393 | 0.579 | 88 / 41 (p 4e-5) | 862 = 0.718 | 285 / 289 / 288 | 0.827 | 0.344 | 0.611 | 115 / 41 (p 3e-9) |

- **The pattern of round 8 holds on every seed.** Thresholds halved or doubled and a mask off by one pixel change
  nothing (0.694–0.700 as trained, 0.782–0.784 with guidance; net −4 to +5 episodes of 1200).
- **Random flips, 10%**: −2.8 points as trained and −3.0 with guidance, now significant (p 0.002, 0.004; one seed in
  round 8: p 0.16, 0.21). With 30% the policy as trained is at the level of the baselines (0.613; rgbdt 0.613,
  p 1) and the guided one at 0.678, still above them (p 0.001, 0.006).
- **A sensor left on where it does not show the glass is the costly error, more so with guidance**: see-through
  episodes 0.548 → 0.400 (flips, 30%), 0.393 (mask missing), 0.196 (always on) as trained and 0.574 → 0.289, 0.344,
  0.093 with guidance; crossover episodes with thermal always on 0.643 → 0.444 and 0.778 → 0.333. With both sensors
  always on the policy is at the level of the baselines in both settings (0.603 and 0.635; p ≥ 0.29).
- **A sensor wrongly off**: a mask shifted up to 2 px (depth off in 38% of the normal steps, thermal in 22%) costs
  5.8 points as trained and 3.9 with guidance, all on normal and crossover episodes (see-through unchanged); with
  three seeds the loss with guidance is significant too (p 2e-4; round 8: p 0.41). Depth always off costs 12.2 and
  16.3 points (0.575 and 0.617).
- **Guidance gains under every error**: +6.0 to +10.2 points in nine of the eleven conditions (p < 1e-5 each), +4.2
  with depth always off (p 0.001) and +3.2 with both always on (p 0.007).
- Against the baselines (Fisher, 1200 episodes per side): as trained the policy is above rgbdt for no error, flips at
  10%, thermal always off, both threshold changes, the 1 px shift and the missing mask (p ≤ 0.025), and not for flips
  at 30%, depth always off, both always on and the 2 px shift (p 0.061–1); it is below rgbd with depth always off
  (0.575 against 0.624, p 0.016). With guidance it is above both baselines in every condition (p ≤ 0.006) except
  depth always off and both always on (p ≥ 0.29).

### A missing mask keeps the last flags (`--route_hold`)

Mask missing in 30% of the frames, three seeds (1200 episodes); "no error + hold" on seed 42 only (400).

| condition | as trained | normal | see-through | crossover | depth on, see-through | guide 3.0 | normal | see-through | crossover |
|---|---|---|---|---|---|---|---|---|---|
| no error | 837 = 0.698 | 0.741 | 0.548 | 0.643 | 0.5% | 936 = 0.780 | 0.840 | 0.574 | 0.778 |
| mask missing, 30% | 790 = 0.658 | 0.735 | 0.393 | 0.579 | 29.8% | 862 = 0.718 | 0.827 | 0.344 | 0.611 |
| mask missing, 30%, hold | 825 = **0.688** | 0.728 | 0.548 | 0.619 | 0.4% | 915 = **0.762** | 0.817 | 0.574 | 0.762 |
| no error, seed 42 | 282 = 0.705 | 0.739 | 0.589 | 0.690 | 0.1% | 311 = 0.777 | 0.829 | 0.600 | 0.786 |
| no error, hold, seed 42 | 284 = 0.710 | 0.748 | 0.578 | 0.738 | 0.4% | 314 = 0.785 | 0.842 | 0.589 | 0.786 |

- **Holding the flags removes the cost of missing masks on see-through and crossover episodes**: as trained 0.658 →
  0.688 (same episodes 47 lost, 82 gained, p 0.003), see-through 0.393 → 0.548 (11 lost, 53 gained, p 1e-7); with
  guidance 0.718 → 0.762 (62 / 115, p 8e-5), see-through 0.344 → 0.574 (13 / 75, p 9e-12), crossover 0.611 → 0.762
  (10 / 29, p 0.003).
- **Against no error**: as trained 0.688 against 0.698 (51 lost, 39 gained, p 0.25); with guidance 0.762 against
  0.780 (58 lost, 37 gained, p 0.040), the difference on normal episodes (0.817 against 0.840; 45 lost, 24 gained,
  p 0.015), where a sensor is off in 2.9% of the decisions (0.7% without errors); with held flags the decisions
  before the first mask of an episode run on rgb alone.
- **Without errors holding changes nothing** (seed 42): 0.710 against 0.705 (6 lost, 8 gained), 0.785 against 0.777
  with guidance (12 / 15).

### Switched composition of the rgbd and rgbt policies (seeds 42 and 43, 800 episodes per row)

| policy | seed 42 / 43 (of 400) | all | normal (620) | see-through (180) | crossover (84) | neither (548) |
|---|---|---|---|---|---|---|
| rgbd | 261 / 251 | 0.640 | 0.798 | 0.094 | 0.774 | 0.788 |
| rgbt | 228 / 177 | 0.506 | 0.508 | 0.500 | 0.262 | 0.540 |
| rgbdt | 239 / 262 | 0.626 | 0.747 | 0.211 | 0.548 | 0.763 |
| mean of rgbd and rgbt, 0.75 / 0.25 (round 8) | 271 / 246 | 0.646 | 0.779 | 0.189 | 0.667 | 0.781 |
| **rgbd where depth shows the glass, rgbt elsewhere** | 296 / 277 | 573 = **0.716** (0.68–0.75) | 0.781 | 0.494 | 0.714 | 0.781 |
| nogate_rgbonly | 273 / 245 | 0.647 | 0.681 | 0.533 | 0.488 | 0.703 |
| vis_route | 282 / 273 | 0.694 | 0.734 | 0.556 | 0.655 | 0.743 |
| vis_route, guide 3.0 | 311 / 305 | 0.770 | 0.827 | 0.572 | 0.762 | 0.832 |
| vis_route on rgb + depth | 270 / 262 | 0.665 | 0.708 | 0.517 | 0.607 | 0.719 |
| vis_route on rgb + depth, guide 3.0 | 310 / 301 | 0.764 | 0.834 | 0.522 | 0.798 | 0.832 |

The rgbd member makes 99.7% of the decisions on normal episodes, 1.0% on see-through ones and 85.8% on crossover
ones.

- **Switching between two plain policies with the same flags is level with `vis_route` as trained**: 0.716 against
  0.694 (Fisher p 0.35; same seeds and episodes 107 lost, 125 gained, p 0.26); normal 0.781 against 0.734 (p 0.064),
  see-through 0.494 against 0.556 (p 0.29). It is above rgbd (p 0.001), rgbdt (p 2e-4), the control (p 0.004), the
  weighted mean of round 8 (p 0.003) and `vis_route` on rgb + depth as trained (0.665, p 0.031).
- **With guidance `vis_route` is above it**: 0.770 against 0.716 (Fisher p 0.016; 129 lost, 86 gained from the
  guided policy to the composition, p 0.004), normal 0.827 against 0.781 (p 0.045); `vis_route` on rgb + depth with
  guidance 0.764 (p 0.035; 130 / 92, p 0.013). The composition has no rgb-only path to guide from, and it needs two
  trained policies at inference.
- Its see-through episodes are at the level of its rgbt member (0.494 against 0.500) and its normal episodes at
  the level of its rgbd member (0.781 against 0.798): the switch hands each scene to the right member, as the
  weighted mean could not (see-through 0.189).

### Conclusions of round 9

33. **The thermal image has a small, measurable part in `vis_route`** (revises conclusion 30, which had one seed):
    switched off at inference the policy loses 2.5 points as trained (0.698 → 0.672, matched p 0.024), and a
    `vis_route` trained on rgb + depth is 3.5 points lower as trained (0.662, matched p 0.009). With guidance the
    totals are 0.8 and 2.1 points lower (0.772 and 0.759 against 0.780, p 0.58 and 0.14), but see-through episodes
    are about 8 points lower without thermal in both cases (0.489 and 0.496 against 0.574, p 0.013 and 0.038):
    thermal is used where depth sees through the glass.
34. **The method does not depend on the thermal image in the policy**: on rgb + depth, routing with guidance reaches
    0.759, above rgbd (0.624) and rgbdt (0.613) at the episode and the seed level, and guidance gains as much (+9.7
    points). The mask of the gate was still predicted from all three images.
35. **The gate-error results of round 8 hold over three seeds** (removes the first caveat of round 8): small errors
    cost nothing; 10% of the flags flipped costs 3 points in both settings, now significant; a sensor on where it
    does not show the glass is the costly error and guidance makes it worse there (see-through 0.09 with both always
    on); with guidance the policy stays above both baselines unless one sensor is forced in every step. The gain
    from guidance is +6 to +10 points except with depth always off (+4.2) and both always on (+3.2).
36. **A missing mask should keep the last decision** (follow-up of conclusion 29): with the flags held, 30% missing
    masks cost 1.0 point as trained (p 0.25) and 1.8 with guidance (p 0.040) instead of 3.9 and 6.2, and see-through
    episodes are back at the level without errors.
37. **Switching between plain rgbd and rgbt policies by the same visibility flag reaches `vis_route` as trained**
    (0.716 against 0.694, p 0.35), and stays below `vis_route` with guidance (0.770, p 0.016). As trained, what
    `vis_route` adds over this baseline is one policy instead of two; its advantage in success comes with guidance.
    The visibility flag itself is what both use.

Caveats: the gate still uses a glass mask predicted from rgb, depth and thermal, so "without thermal" refers to the
policy's input only; the random gate errors are one draw per seed and independent between decision steps; the
composition has two seeds (rgbt has no third) and its rgbt member differs by 51 episodes between them; holding the
flags without errors was run on one seed; "matched" pairs runs by seed number; seed-level tests have three values
per side; p values are not corrected for the number of comparisons.

### Next steps (proposals, not run)

1. A gate whose mask does not need the thermal image (segmentation from rgb + depth), to test the rgb + depth system
   end to end.
2. The third seed of rgbt and rgb (2 trainings), for the composition baselines.
3. Open from rounds 7 and 8: the control with visibility-independent single-sensor dropout; the control's own
   guidance scale; partially visible targets.

## Round 10: a routing gate that is trained (11/18 08:03 – 15:11 KST)

The gate of `vis_route` is a rule: a sensor is on when the step of its image at the predicted glass mask is above a
fixed threshold (depth 15, thermal 30). Round 10 replaces the two thresholds by a small network that is trained, in
two ways, and checks each against a criterion fixed before the runs.

### Runs

Two trainings and four evaluations, seed 42 only; all six jobs finished at the first attempt.

- `L0_bench_rgbdt_500demos_learn_labels`: the gate is a classifier of "this sensor shows the glass", trained with a
  cross-entropy on the episode labels of the simulator (depth: not a see-through episode; thermal: not a crossover
  episode). During training the sensors of the policy are switched by these labels, at inference by the gate.
- `L0_bench_rgbdt_500demos_learn_heldout`: no labels. Every 5th demo is left out of the policy's loss: 93 of the 475
  training demos (25 of the 500 are the validation split in every run), so the policy is fitted to 382 demos.
  On those demos the denoising loss is evaluated, without gradient, with each sensor on and off (same noise and
  timestep), and the gate is moved toward the state with the lower loss: loss_gate = Σ_m p_m (L_on,m − L_off,m).
  The gate loss starts after 2000 steps. During training the sensors of the policy are switched by decisions sampled
  from the gate, each state kept with probability at least 0.1 (`route_explore`).
- Both: the `vis_route` command of rounds 3–5 with
  `policy.obs_encoder.dropout_mode=none +policy.obs_encoder.route=learned`
  `+policy.obs_encoder.route_rgb_only_p=0.1 policy.gate_loss_weight=0`
  and `+policy.gate_train=labels`, or
  `+policy.gate_train=heldout +task.dataset.gate_split=5 +policy.obs_encoder.route_explore=0.1`; epoch 100 without
  interruption; the benchmark evaluation at 50 episodes per test set, then 200 per test set in `n200/`, as trained
  and with `--guidance 3.0` (`__guide30`), all with `--save_w`.

The gate: p_m = sigmoid(MLP_m(cue)) for depth and thermal, on when p_m > 0.5 at inference. Its input is the
10-entry cue of `vis_route` with the two thresholded flags set to zero, so it sees the steps at the glass mask but
not the rule's decisions. As for `vis_route`, the mask comes from the rgb + depth + thermal segmentation network.
The decision is hard, so the imitation loss gives the gate no gradient; it is trained by the gate loss alone.

Code added, all opt-in (`python -m policy.test_gating` passes; runs without the new options are unchanged):
`policy/gated_obs_encoder.py` (`route=learned`, `route_explore`), `policy/gated_policy.py` (`gate_train`,
`gate_start_step`, diagnostics), `policy/glass_dataset.py` (`gate_split`).

Fixed before the runs:

- A variant passes when (i) the gate's decisions on the test episodes agree with the visibility flags in at least
  95% of the decisions, for each sensor, and (ii) its success is not more than 3 points below `vis_route` of the same
  seed: at least 0.675 as trained (0.705) and 0.747 with guidance 3.0 (0.777).
- Only a variant that passes is trained on seeds 43 and 44.
- Fixed before the evaluations: agreement is measured per decision step between the gate's decision and the flag of
  the rule in the same step (both stored by `--save_w`), over the 400 test episodes of a run.

### Training

Means over the 500 steps before each line of the training log (410 steps per epoch). "vis" / "not vis": samples of
episodes where the sensor shows / does not show the glass (episode labels; used for these diagnostics in both runs).

`learn_labels`:

| step | denoising loss | gate accuracy | p depth, vis | p depth, not vis | p thermal, vis | p thermal, not vis |
|---|---|---|---|---|---|---|
| 2500 | 0.0203 | 0.967 | 0.908 | 0.067 | 0.874 | 0.834 |
| 10000 | 0.0095 | 0.980 | 0.999 | 0.001 | 0.955 | 0.561 |
| 20000 | 0.0054 | 0.988 | 1.000 | 0.000 | 0.970 | 0.386 |
| 30000 | 0.0028 | 0.990 | 1.000 | 0.000 | 0.974 | 0.340 |
| 41000 | 0.0015 | 0.990 | 1.000 | 0.000 | 0.975 | 0.325 |

`learn_heldout` (Δ = loss with the sensor on − loss with it off, on the held-out demos; negative: the sensor helps):

| step | denoising loss | Δ depth, vis | Δ depth, not vis | Δ thermal, vis | Δ thermal, not vis | p depth, vis | p depth, not vis | p thermal, vis | p thermal, not vis |
|---|---|---|---|---|---|---|---|---|---|
| 2500 | 0.0222 | −0.0006 | −0.0001 | −0.0003 | −0.0000 | 0.504 | 0.503 | 0.501 | 0.500 |
| 10000 | 0.0110 | −0.0013 | −0.0002 | −0.0007 | +0.0006 | 0.963 | 0.727 | 0.702 | 0.600 |
| 20000 | 0.0058 | −0.0021 | +0.0002 | −0.0015 | +0.0017 | 0.996 | 0.825 | 0.958 | 0.896 |
| 30000 | 0.0030 | −0.0030 | +0.0007 | −0.0009 | +0.0061 | 0.997 | 0.722 | 0.971 | 0.895 |
| 41000 | 0.0016 | −0.0035 | +0.0008 | +0.0002 | +0.0100 | 0.997 | 0.682 | 0.964 | 0.857 |

- **Labels**: the depth gate separates the two cases within 10,000 steps. The thermal gate does not: its mean
  probability on the samples of crossover demos (32 of the 475 training demos) is still 0.33 at the end, and at inference it
  leaves thermal on in 16% of the crossover decisions (table below).
- **Held-out demos**: the loss differences have the sign the labels would give (means from step 20000 on: depth
  −0.0030 where visible and +0.0005 where not, thermal −0.0010 and +0.0054), so the held-out loss does carry the
  information. They are small while the gate is formed (at most 0.002 up to step 20000, against a denoising loss of
  0.006–0.02; of the 93 held-out demos 7 are crossover and 21 see-through ones) and largest at the end, when the
  learning rate of the cosine schedule is small (below a quarter of its peak from step 28,000 on, below a twentieth
  from step 36,000 on). The gate moved
  toward "on" for both sensors in every case during the first 20,000 steps and came back only slowly (depth where not
  visible 0.82 → 0.68, thermal where not visible 0.91 → 0.86): it ends as "both on".

### Results (seed 42, 400 episodes per row)

50 episodes per test set at epoch 100: `learn_labels` (a) 0.72, (g) 0.62; `learn_heldout` 0.54 / 0.50 (`vis_route`
0.72 / 0.66). 200 episodes per test set, the same test episodes in every row:

| run | guidance | all | normal (310) | see-through (90) | crossover (42) | neither (274) | criterion |
|---|---|---|---|---|---|---|---|
| rgbd | – | 261 = 0.652 | 0.816 | 0.089 | 0.857 | 0.796 | |
| rgbdt | – | 239 = 0.598 | 0.706 | 0.222 | 0.571 | 0.715 | |
| nogate_rgbonly | – | 273 = 0.682 | 0.710 | 0.589 | 0.500 | 0.737 | |
| vis_route, both always on | – | 234 = 0.585 | 0.703 | 0.178 | 0.381 | 0.737 | |
| vis_route | – | 282 = 0.705 | 0.739 | 0.589 | 0.690 | 0.745 | |
| **learn_labels** | – | 279 = **0.698** (0.65–0.74) | 0.761 | 0.478 | 0.595 | 0.785 | ≥ 0.675: met |
| **learn_heldout** | – | 221 = **0.552** (0.50–0.60) | 0.552 | 0.556 | 0.476 | 0.558 | ≥ 0.675: not met |
| nogate_rgbonly | 3.0 | 273 = 0.682 | 0.729 | 0.522 | 0.500 | 0.748 | |
| vis_route, both always on | 3.0 | 252 = 0.630 | 0.784 | 0.100 | 0.262 | 0.847 | |
| vis_route | 3.0 | 311 = 0.777 | 0.829 | 0.600 | 0.786 | 0.832 | |
| **learn_labels** | 3.0 | 292 = **0.730** (0.68–0.77) | 0.784 | 0.544 | 0.667 | 0.803 | ≥ 0.747: not met |
| **learn_heldout** | 3.0 | 255 = **0.637** (0.59–0.68) | 0.665 | 0.544 | 0.476 | 0.686 | ≥ 0.747: not met |

Agreement of the gate's decisions with the flags of the rule in the same decision step:

| run | guidance | decisions | depth | thermal | both | depth on where not visible | thermal on where not visible | sensor off where visible | criterion |
|---|---|---|---|---|---|---|---|---|---|
| learn_labels | – | 5940 | 0.994 | 0.981 | 0.977 | 1.4% | 15.7% | 0.3% / 0.2% | ≥ 0.95 each: met |
| learn_labels | 3.0 | 5980 | 0.994 | 0.982 | 0.977 | 2.1% | 15.7% | 0.1% / 0.1% | met |
| learn_heldout | – | 6000 | 0.874 | 0.899 | 0.794 | 53.9% | 92.5% | 0% / 0% | not met |
| learn_heldout | 3.0 | 6000 | 0.870 | 0.896 | 0.790 | 54.7% | 93.2% | 0% / 0% | not met |

**Verdict by the fixed criterion: neither variant passes, and neither is trained on seeds 43 and 44.**
`learn_labels` meets three of the four parts (agreement in both settings, success as trained) and misses the success
limit with guidance (0.730 against 0.747). `learn_heldout` misses all four.

- **`learn_labels` as trained is level with the rule**: 0.698 against `vis_route` 0.705 (Fisher p 0.88; same
  episodes 38 lost, 35 gained, p 0.82), above rgbdt (0.598, p 0.004) and above the same kind of policy with both
  sensors always on (0.585, p 0.001); against the control 0.682 (p 0.70) and rgbd 0.652 (p 0.20). By group it is
  0.761 / 0.478 / 0.595 against 0.739 / 0.589 / 0.690 for `vis_route` (normal p 0.58, see-through p 0.18, crossover
  p 0.50).
- **With guidance it is 4.8 points below the rule**: 0.730 against 0.777 (Fisher p 0.14; same episodes 45 lost, 26
  gained, p 0.032). Guidance adds 3.2 points to `learn_labels` (26 lost, 39 gained, p 0.14) against 7.2 to
  `vis_route` at this seed (23 / 52, p 0.001). It stays above "both always on" with guidance (0.630, p 0.003);
  against the guided control 0.682, p 0.16.
- **Where the learned classifier differs from the rule**: thermal stays on in 15.7% of the decisions where the rule
  has it off; on crossover episodes the thermal decision agrees with the rule in 87% of the decisions (thermal on in
  16%, visible by the rule in 4%). Depth agrees in 99% in every group. Whether the smaller gain from guidance comes
  from these thermal decisions (conclusions 29 and 35: a sensor wrongly on costs more with guidance), from training
  the policy
  with episode labels instead of per-frame flags, or from the seed is not separated by this run.
- **`learn_heldout` ends with a gate that is mostly on and a weaker policy**: thermal is on in 92% of the decisions
  where the rule has it off and depth in 54% (on see-through episodes depth is on in 52% of the decisions, visible
  by the rule in 0.4%). Success is 0.552, below `vis_route` (p 1e-5; 85 lost, 24 gained) and the control (p 2e-4),
  level with rgbdt (0.598, p 0.22) and with `vis_route` with both sensors always on (0.585, p 0.39). The loss is on
  the normal episodes (0.552 against 0.739; rgbdt 0.706), where its gate agrees with the rule (depth 0.986); the
  see-through episodes are at the level of `vis_route` (0.556 against 0.589). Its policy was trained on 382 demos
  instead of 475 and with sensors switched off at random in at least 10% of the samples each, independent of
  visibility; which of the two lowers the normal episodes is not separated. Guidance adds 8.5 points (25 lost, 59
  gained, p 3e-4).
- Per test set: `learn_labels` (a) 0.725, (g) 0.670, with guidance 0.750 / 0.710; `learn_heldout` 0.585 / 0.520 and
  0.645 / 0.630 (`vis_route` 0.735 / 0.675 and 0.780 / 0.775). Failure outcomes over 400 episodes: `learn_labels`
  missed grasp 55, dropped 41, wrong height 15, other 8, wrong object 2; `learn_heldout` 87 / 53 / 24 / 12 / 3.

### Conclusions of round 10

38. **The two thresholds of the rule can be replaced by a classifier trained on the simulator's visibility labels
    without a measurable loss as trained**: 0.698 against 0.705, with 99.4% (depth) and 98.1% (thermal) of its
    decisions equal to the rule's. With guidance 3.0 it is 4.8 points below the rule (0.730 against 0.777, matched
    p 0.032) and misses the limit fixed before the run. One seed.
39. **A gate trained without labels, from the loss on held-out demos, did not work in this form**: the loss
    differences point the right way and grow during training, but the gate saturates at "on" before they are large,
    and the policy trained next to it is at the level of rgbdt (0.552).
40. **The supervision of the gate is still the simulator's**: the labels are episode labels of the simulator, and the
    input of both gates is the cue at the glass mask of the segmentation network trained with simulator masks. What
    round 10 shows is that the two thresholds need not be set by hand, not that visibility can be found without
    labels.

Caveats: one seed and one run per variant; the guidance scale 3.0 was selected for `vis_route` and not for the
learned gates; the agreement is measured against the rule, not against ground-truth visibility per frame; the
held-out variant differs from `vis_route` in three things at once (the gate, 382 instead of 475 demos for the policy,
random switching during training); p values are not corrected for the number of comparisons.

### Next steps (proposals; the first was run as round 11)

1. Fit the gate of `learn_heldout` after training: policy frozen, a fresh gate trained on the held-out demos with
   the same loss difference and its own learning rate; evaluations only. This tests whether the label-free signal
   reaches the rule's decisions when it is not tied to the policy's schedule.
2. `learn_labels` on seeds 43 and 44 (2 trainings), to tell the 4.8 points with guidance from seed variation. Not
   covered by the rule fixed before the run; needs a decision.
3. For a label-free gate trained jointly: normalize the loss difference per sensor (its scale changes tenfold during
   training), start the gate later, or add a cost per sensor that is on.

## Round 11: the gate of `learn_heldout` fitted after training (11/18 23:50 – 11/19 01:02 KST)

Proposal 1 of round 10: is the label-free signal of `learn_heldout` wrong, or only the way it was used during
training? The policy of that run is frozen and a new gate is fitted to the same loss difference afterwards.

### Runs

- **Fit**: `python policy/fit_gate.py -c data/runs/L0_bench_rgbdt_500demos_learn_heldout/checkpoints/latest.ckpt`
  `-o results/hard_scenario/fusion_bench/gate_fit/L0_bench_rgbdt_500demos_learn_heldout`. The EMA weights of epoch
  100, all frozen. For each of the 5128 samples of the 93 held-out demos the denoising loss is computed in the four
  sensor states (depth on / off × thermal on / off), with the same noise and timesteps in the four states, averaged
  over 8 draws. A fresh gate (p = 0.5 for every input) is trained on these cached losses with the loss of
  `gate_train=heldout`, Σ_m p_m (L_on,m − L_off,m): 2000 full-batch Adam steps, learning rate 1e-3, seed 0. No
  labels are used. 100 seconds on one GPU. Output: `gate_fit/<run>.pt` (the gate, 9 kB) and `.json` (diagnostics).
- **Four evaluations** of the same policy on the test episodes of seed 42 (200 per test set, as trained and with
  guidance 3.0), all finished at the first attempt: `__postfit[_g30]` with the fitted gate (`--gate_state`), and
  `__flags[_g30]` with the sensors switched by the thresholded flags of the rule instead of a gate (`--route flags`),
  as the reference for what this policy reaches with the decisions of `vis_route`.

Code added, all opt-in (`python -m policy.test_gating` passes): `policy/fit_gate.py`; `policy/eval.py`
(`--gate_state`, `--route flags`); `policy/gated_obs_encoder.py` (`route_eval=flags` for `route=learned`).

Fixed before the fit:

- The settings above; one fit, nothing selected.
- **Criterion A (decisions)**: on the test episodes the fitted gate agrees with the flags of the rule in at least 95%
  of the decisions, for each sensor (the criterion of round 10).
- **Criterion B (success)**: with the fitted gate the policy is not more than 3 points below the same policy switched
  by the flags of the rule, as trained and with guidance. The limits of round 10 (0.675 and 0.747) are reported too.
- Expectation written down before the fit: depth is likely to pass; thermal may end mostly off, because the logged
  loss difference where thermal shows the glass was close to zero (round 10, +0.0002 at the end of training).

### The fit on the held-out demos

Mean denoising loss by state: both on 0.02427, thermal off 0.02404, depth off 0.02678, both off 0.02699.
"Rule on / off": the samples where the flag of the rule is on / off at the last observation step.

| gate | gate loss | depth: on (rule 77.0%) | agrees with rule | on where rule off | off where rule on | thermal: on (rule 73.2%) | agrees with rule | on where rule off | off where rule on |
|---|---|---|---|---|---|---|---|---|---|
| of the checkpoint (trained jointly) | −0.00261 | 90.9% | 0.861 | 60.3% | 0.0% | 99.4% | 0.738 | 97.9% | 0.0% |
| fresh (p = 0.5) | −0.00136 | | | | | | | | |
| **fitted after training** | **−0.00400** | 82.0% | **0.951** | 21.5% | 0.0% | 44.0% | 0.530 | 33.3% | 52.0% |

Loss difference (sensor on − off) under the fitted gate: depth −0.0037 where the rule has it on and +0.0013 where
off; thermal −0.0004 where on and +0.0019 where off (by episode label: −0.0004 on demos without crossover, +0.0085
on crossover demos, where the fitted gate has thermal on in 4.9% of the steps).

### Test episodes (seed 42, 400 episodes per row, the same policy in the first three rows)

Decisions against the flags of the rule in the same decision step, as trained (with guidance within 1 point, except
thermal on where not visible: 10.1%):

| gate | depth agrees | depth on where not visible | depth off where visible | thermal agrees | thermal on where not visible | thermal off where visible | thermal agrees, crossover episodes |
|---|---|---|---|---|---|---|---|
| trained jointly (round 10) | 0.874 | 53.9% | 0.0% | 0.899 | 92.5% | 0.0% | 0.119 |
| **fitted after training** | **0.964** | 13.1% | 0.7% | **0.614** | 6.3% | 42.6% | 0.914 |

| gate | as trained | normal (310) | see-through (90) | crossover (42) | guide 3.0 | normal | see-through | crossover |
|---|---|---|---|---|---|---|---|---|
| trained jointly (round 10) | 221 = 0.552 | 0.552 | 0.556 | 0.476 | 255 = 0.637 | 0.665 | 0.544 | 0.476 |
| **fitted after training** | 218 = **0.545** (0.50–0.59) | 0.542 | 0.556 | 0.476 | 249 = **0.623** (0.57–0.67) | 0.642 | 0.556 | 0.500 |
| flags of the rule | 221 = 0.552 | 0.552 | 0.556 | 0.452 | 256 = 0.640 | 0.661 | 0.567 | 0.524 |
| `vis_route` (its own policy) | 282 = 0.705 | 0.739 | 0.589 | 0.690 | 311 = 0.777 | 0.829 | 0.600 | 0.786 |

**Verdict by the fixed criteria.** A is not met: depth 0.964 (0.966 with guidance) passes, thermal 0.614 (0.618)
does not, as expected before the fit. B is met: −0.8 points as trained (Fisher p 0.89; same episodes 12 lost, 9
gained) and −1.8 with guidance (p 0.66; 26 lost, 19 gained). The limits of round 10 are missed by every gate on
this policy, the flags of the rule included.

- **Fitted after training, the label-free loss finds the depth decision of the rule**: 96.4% agreement on the test
  episodes against 87.4% for the gate trained jointly; depth is left on where the rule has it off in 13% of the
  decisions instead of 54% (on see-through episodes on in 12% instead of 52%), and almost never switched off where
  the rule has it on (0.7%).
- **For thermal the loss says "off" far more often than the rule**: on in 52% of the decisions (the rule: 89%). It
  does switch thermal off where it does not show the glass (left on in 6% instead of 93%; agreement on crossover
  episodes 0.91 instead of 0.12), but also in 43% of the decisions where it does. On the held-out demos thermal
  lowers the loss of this policy by 0.0004 where the rule has it on, a tenth of what depth does (0.0037).
- **Criterion B does not discriminate on this policy.** Its success is the same with all three gates: 0.552, 0.545
  and 0.552 as trained, 0.637, 0.623 and 0.640 with guidance (largest difference 26 lost against 19 gained), on
  see-through episodes 0.556 whether depth is on in 52%, 12% or 0.1% of the decisions.
- **The policy, not the gate, keeps `learn_heldout` below `vis_route`**: switched by the flags of the rule it is at
  0.552 against 0.705 (p 1e-5; 85 lost, 24 gained) and 0.640 against 0.777 with guidance (p 2e-5), the difference on
  normal episodes (0.552 against 0.739, 0.661 against 0.829) and on crossover ones (0.452 against 0.690, p 0.047).
  It was fitted to 382 demos instead of 475 and with each sensor switched at random in at least 10% of the samples.
- Guidance adds 8.5, 7.8 and 8.8 points with the three gates (p ≤ 3e-4 each).

### Conclusions of round 11

41. **The label-free signal was right for depth; the joint schedule was the problem** (follow-up of conclusion 39):
    the same loss difference, fitted after training with the policy frozen, agrees with the depth flag of the rule
    in 96% of the test decisions, without labels and in 100 seconds.
42. **For thermal the signal does not reproduce the rule, because this policy gains almost nothing from thermal**:
    the fitted gate switches thermal off on crossover scenes and in 43% of the decisions where the rule keeps it on.
    This fails the agreement criterion; it is in line with the small part of thermal found in round 9
    (conclusion 33).
43. **A gate fitted after training does not help a policy that was trained without it**: this policy reaches 0.55
    (0.62–0.64 with guidance) with any of the three gates, 15 points below `vis_route`. Whether a label-free gate
    reaches the success of the rule needs the policy trained with the fitted gate on all demos; that was not run.

Caveats: one seed and one fit (the gate loss was still falling slowly at step 2000); the 5128 samples are
overlapping windows of 93 demos; agreement is measured against the rule, not against ground-truth visibility; the
loss differences are those of a policy trained with random switching and say what this policy does with a sensor,
not what a policy trained with the gate would do; p values are not corrected for the number of comparisons.

### Next steps (proposals, not run)

1. Two stages: train the policy again on all 475 training demos with the fitted gate fixed (as `vis_route` is
   trained with the rule), seed 42 first (1 training, 2 evaluations), and compare with `vis_route`.
2. `learn_labels` on seeds 43 and 44 (open from round 10).
3. Keep the rule as the gate of the method and report rounds 10 and 11 as what a learned gate reaches.

## Round 12: trained gates on three seeds, two-stage training (started 11/19 02:18 KST)

### Plan (written before the results)

Requirement for the gate from here on: it is decided by training, not by thresholds set by hand; supervision by the
labels of the simulator is allowed.

Trainings (L0_bench, rgbdt, 500 demos, 101 epochs, the `vis_route` settings with `route=learned`):

- `learn_labels_seed43`, `learn_labels_seed44`: the label-trained gate of round 10, unchanged, on the other two seeds.
- `learn_twostage` (seed 42): the second stage of round 11. The gate fitted after training
  (`gate_fit/L0_bench_rgbdt_500demos_learn_heldout.pt`) is loaded and not trained (`policy.gate_train=frozen`); the
  policy is trained on all 475 training demos with the sensors switched by that gate (p > 0.5) and 10% rgb-only
  samples, as `vis_route` is trained with the rule.
- `learn_labels_bal` (seed 42): `learn_labels` with the two label classes of each sensor weighted equally in every
  batch (`policy.gate_balance=true`), aimed at the thermal decision on crossover scenes (left on in 16% of the
  decisions in round 10; 32 of the 475 training demos are crossover demos).

Form of the gate for the two-stage run, checked without labels before the training
(`fit_gate.py --select --draws 32`, `gate_fit/L0_bench_rgbdt_500demos_learn_heldout_select.json`): a linear gate or
the MLP, 300 to 10,000 steps, weight decay 0 or 0.01, compared by the gate loss on demos left out of the fit (5 folds
over the 93 held-out demos). The lowest left-out loss is the MLP at 3000 steps (−0.003451; linear at 10,000 steps
−0.003417), with the decisions of the gate of round 11 (on the left-out demos depth agrees with the rule in 95.7%
of the steps, thermal in 52.1%); in every setting thermal agrees in 46–52%. The thermal decision of round 11 is
therefore not an artefact of the fit, and the gate of round 11 is used unchanged.

Evaluations: 200 episodes per test set, as trained and with guidance 3.0 (the scale selected for `vis_route` in
round 6; no other scale is tried on the test episodes). Diagnostics: `learn_labels` of seed 42 switched by the flags
of the rule (`__flags[_g30]`), and thermal off at inference (`--drop thermal`) for the label-trained gates.

Criteria:

- Label-trained gate over three seeds (1200 episodes): decisions agree with the rule in at least 95% for each
  sensor, and success is not more than 3 points below `vis_route` on the same seeds (0.698 and 0.780): at least
  0.668 as trained and 0.750 with guidance.
- `learn_twostage` and `learn_labels_bal` at seed 42: the limits of round 10 (0.675 as trained, 0.747 with
  guidance). A variant that meets them is run on seeds 43 and 44.

Deviation from the plan (11/19 03:02 KST, before any result): `learn_labels_bal` was started at 02:42 and stopped at
epoch 1. With four trainings the free memory fell to 17 GB (each training holds 17–24 GB here), next to another
long training on the machine; three trainings and one evaluation at a time are run instead. Its epoch-0 checkpoint
was removed. The variant is not run in this round.

## Round 13: the glass location as an input (planned 11/19 04:45 KST)

### What was measured first (no policy involved)

`policy/check_glass_mask.py`, output in `glass_loc/mask_check.txt`: the glass mask of the segmentation network
(`mix_ws`, the one behind the visibility flags) against the true mask on the segmentation test frames of L0_bench
(549 frames per test set), with all sensors, without thermal, without depth and with rgb alone. A sensor is left out
the way the network was trained with missing sensors.

| frames (test set g, held-out conditions) | all sensors | without thermal | rgb only |
|---|---|---|---|
| depth and thermal show the glass (371) | 100% within 4 px, median 0.48 px | 100%, 0.48 px | 93.8%, 0.70 px |
| depth blind, thermal shows it (111) | 98.2%, 0.65 px | 100%, 0.61 px | 100%, 0.76 px |
| thermal blind, depth shows it (55) | 100%, 0.44 px | 100%, 0.46 px | 98.2%, 0.92 px |
| both blind (12 frames, 1 episode) | 100%, 0.44 px | 100%, 0.33 px | 100%, 0.30 px |

(share of frames whose mask centroid is within 4 px of the true one, and the median distance; the frame is 96 px
wide and the glass mask about 4 x 8 px.) Test set (a) and the demo frames give the same picture: a mask is found in
every frame (with rgb alone in all but 2 of the 31,120 demo frames), and its centre does not depend on thermal or
depth. On 8000 demo frames the centroid of the rgb-only
mask is within 4 px of the stored all-sensor mask in 98.9–100% of the frames of every scene group.

So the system knows where the glass is in every scene type, from rgb alone, while the policies succeed in about half
of the see-through scenes (`learn_labels`, seed 42: 43/90 = 0.478 as trained; normal scenes 236/310 = 0.761). Two
consequences are tested in this round: whether telling the policy the location closes that gap, and whether depth
and thermal still add anything once the location is given.

### Plan (written before the results)

New observation `glass_loc` (`policy/obs_util.py` `glass_location`): [found, centre x, centre y, width, height] of
the glass mask of the frame, as fractions of the frame, appended to the conditioning like the joint angles. Training
reads it from stored masks, evaluation predicts the mask per frame. Policies without the key are unchanged.

Trainings (seed 42, 101 epochs, the settings of their base runs):

- `learn_labels_loc`: `learn_labels` plus `glass_loc` from the all-sensor mask (the one of the visibility cues).
- `rgb_loc`: the plain rgb policy (`L0_bench_rgb_500demos`) plus `glass_loc` from the mask of the network restricted
  to rgb, in training (`data/fusion_bench_masks_L0_bench_500_rgb.npy`, made with `policy/make_glass_masks.py ... rgb`)
  and at inference. Depth and thermal are used nowhere in this run.

Evaluations, 200 episodes per test set: both runs as trained; `learn_labels_loc` also with guidance 3.0 (the scale
of round 6) and under three conditions at inference, each as trained and with guidance:

- `--seg_mods rgb,depth`: the mask (gate cues and location) without thermal;
- `--drop thermal --seg_mods rgb,depth`: thermal removed from the whole system;
- `--route rgb_only --seg_mods rgb`: the same network with rgb and the rgb-only location, depth and thermal off.

Criteria:

- A, the location helps: `learn_labels_loc` against `learn_labels` of seed 42 on the same 400 episodes, with
  guidance 3.0 (the operating point of the earlier rounds): higher success with a matched sign test p < 0.05. The
  see-through episodes (90) and the comparison as trained are reported next to it. The run also has to meet the
  limits of rounds 10 and 12 for a seed-42 variant (0.675 as trained, 0.747 with guidance); if it meets them and A
  holds, it is run on seeds 43 and 44.
- B, thermal with the location given: thermal removed from the whole system against the full system on the same
  episodes; thermal contributes if the matched sign test gives p < 0.05 over all episodes or over the see-through
  episodes (both reported, as trained and with guidance).
- C, the other sensors with the location given: `rgb_loc` against `learn_labels_loc` as trained on the same test
  episodes. If `rgb_loc` is not below it (matched sign test p >= 0.05, or above), depth and thermal are not needed in
  this benchmark once a segmentation network gives the location, and the claim of the fusion policy has to change.
  The result is reported either way; the rgb-only condition of `learn_labels_loc` is the same question asked of
  one network.

The two trainings need 11 GB of disk and are started when that space is available.

## Files

- `L0_bench_rgbdt_500demos_<tag>/`: `train_logs.json.txt`, `curve.json`, and per test set `eval_log.json` and
  `episodes.json` (with the gate weights `w` and cues `cue` per decision step for gated runs, `eval.py --save_w`).
- `n200/<run>/`: the 200-episode re-evaluations, benchmark baselines included.
- `summary_table.md`: `policy/summarize_runs.py table` of the runs here; `seg_summary.md`: the `mix_ws` segmentation summary.
- Code added for round 2, all opt-in with the default path unchanged (cue outputs bit-identical on real frames; CPU tests
  in `policy/test_gating.py`): `policy/obs_util.py` (`depth_object` cue, `cue_opts`), `policy/glass_dataset.py`,
  `policy/gated_obs_encoder.py` (6- or 7-entry cue), `policy/gated_policy.py` (`gate_depth_target`, training-log
  diagnostics `w_depth_st` / `w_thermal_xo`), `policy/make_depth_ref.py` and `policy/config/depth_ref_front_cam.npy`;
  `scripts/run_experiment.sh` knobs `TRAIN`, `EVAL_EXTRA`, `TOPK`; `scripts/train_fusion_policy.sh`.
- Rounds 3–5 results: `L0_bench_rgbdt_500demos_{gate_nohole_seed43,static_nohole,vis_route,vis_route_seed43,vis_soft,gate_vis}/`
  and their `n200/` re-evaluations, plus `n200/<vis_route run>__{all_on,guide15,latch,depth_first}/`. `episodes.json`
  of the routed runs has the weights `w` and the 10-entry cue per decision step.
- Code added for rounds 3–5, all opt-in with the default path unchanged (encoder outputs bit-identical on fixed inputs
  before and after; `python -m policy.test_gating`): `policy/glass_seg.py` and `policy/make_glass_masks.py` (glass
  masks), `policy/obs_util.py` (`visibility_cues`, `vis_thr`, `vis_raw`), `policy/glass_dataset.py` (mask array, cue
  normalizer), `policy/gated_obs_encoder.py` (`route=visible|soft`, `route_rgb_only_p`, `route_tau`, `gate_norm`,
  `route_eval`), `policy/gated_policy.py` (`route_latch`, `guidance`, diagnostics), `policy/eval.py` (`--route`,
  `--route_latch`, `--guidance`). The mask array and checkpoints are not in git (`data/`).
- Round 6 results: `L0_bench_rgbdt_500demos_{nogate_rgbonly,vis_route_seed44}/` and their `n200/` evaluations (as
  trained, `__guide15`, `__guide30`), `n200/<vis_route run>__guide30/`, `n200/L0_bench_rgbdt_500demos_vis_soft__{guide15,guide30}/`,
  and `val100/<vis_route run>__g{100,125,150,200,300}/` (validation seeds 200000–200099). No code was added.
- Round 7 results: `L0_bench_rgbdt_500demos_{nogate_rgbonly_seed43,nogate_rgbonly_seed44,nogate_allon}/` and their
  `n200/` evaluations (the control seeds as trained, `__guide15`, `__guide30`; `nogate_allon` as trained),
  `val100/<vis_route run>__g{400,500,700}/`, and `guidance_probe/` (`<run>__{val,train}.json` summaries and `.npz`
  per-sequence values of the first pass, `*_k4.*` of the pass with four noise draws). Code added:
  `policy/analyze_guidance.py` (analysis only; nothing in training or evaluation changed).
- Round 8 results: `L0_bench_{rgbd,rgbdt}_500demos_seed44/` (plain baselines, third seed) and their `n200/`
  evaluations; `n200/L0_bench_rgbdt_500demos_vis_route__{flip10,flip30,doff,toff,thr050,thr200,mshift1,mshift2,mdrop30}[_g30]/`
  and `__allon_g30/` (`eval_log.json` records `gate_error`, `vis_thr`, `mask_noise`); `val100/compose_w{25,50,75}[_seed43]/`
  and `n200/compose_{w75,poe}[_seed43]/` (`eval_log.json` records the member checkpoints and weights). Code added,
  all opt-in: `policy/eval.py` (`--gate_error`, `--vis_thr`, `--mask_noise`), `policy/gated_policy.py` (`gate_error`),
  `policy/obs_util.py` (`degrade_mask`, `mask_noise`), `policy/compose_policy.py`, `policy/eval_compose.py`, tests in
  `policy/test_gating.py`.
- Round 9 results: `L0_bench_rgbd_500demos_vis_route[_seed43|_seed44]/` (`vis_route` on rgb + depth) and their `n200/`
  evaluations (as trained, `__guide30`); `n200/L0_bench_rgbdt_500demos_vis_route_seed{43,44}__<tag>[_g30]/` with the
  tags of round 8, `__allon_g30/` and `n200/L0_bench_rgbdt_500demos_vis_route_seed44__all_on/`;
  `n200/<vis_route run>__mdrop30_hold[_g30]/` (three seeds) and `n200/L0_bench_rgbdt_500demos_vis_route__hold[_g30]/`;
  `n200/compose_switch[_seed43]/`. Code added, all opt-in: `policy/obs_util.py` (cues without a thermal image),
  `policy/gated_policy.py` (`route_hold`), `policy/eval.py` (`--route_hold`), `policy/compose_policy.py`
  (`switch_key`), `policy/eval_compose.py` (`--switch`), tests in `policy/test_gating.py`.
- Round 10 results: `L0_bench_rgbdt_500demos_learn_{labels,heldout}/` and their `n200/` evaluations (as trained,
  `__guide30`). Code added, all opt-in: `policy/gated_obs_encoder.py` (`route=learned`, `route_explore`,
  `route_force`), `policy/gated_policy.py` (`gate_train=labels|heldout`, `gate_start_step`),
  `policy/glass_dataset.py` (`gate_split`), tests in `policy/test_gating.py`.
- Round 11 results: `gate_fit/L0_bench_rgbdt_500demos_learn_heldout.{pt,json}` (the gate fitted after training and
  its diagnostics) and `n200/L0_bench_rgbdt_500demos_learn_heldout__{postfit,flags}[_g30]/` (`eval_log.json` records
  `gate_state` and `route`). Code added, all opt-in: `policy/fit_gate.py`, `policy/eval.py` (`--gate_state`,
  `--route flags`), `policy/gated_obs_encoder.py` (`route_eval=flags`), tests in `policy/test_gating.py`.
