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

### Next steps (proposals, not run)

1. Separate guidance from the gate: an rgbdt policy trained with the 10% rgb-only samples and no routing, evaluated with
   the same guidance (1 training run + evaluations).
2. Guidance scale chosen on validation episodes with seeds outside the test range, and the same guidance on `vis_soft`.
3. A third seed of `vis_route`.
4. Partially visible targets (intermediate thermal gaps, partial see-through) to test 0 / 1 against continuous weights
   where they can differ; this needs new test conditions and is outside the fixed benchmark.
5. Learned attention weighting with a third seed, as proposed above.

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
