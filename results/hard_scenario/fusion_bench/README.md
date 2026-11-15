# Fusion methods on L0_bench

The two fusion methods of the project, applied to the benchmark of `results/hard_scenario/benchmark/` (`L0_bench`,
confirmed 11/15: rgb 0.38, rgbd 0.625, rgbdt 0.605) and compared with its plain Diffusion Policy runs, followed by four
ablations of the cue-gating run. Run on the `exp/fusion-bench` branch (exp/demo-scale + `policy/cue-gating` +
`policy/learned-attention-weighting`), 11/15 17:18 – 11/16 07:00 KST.

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

## Next steps (proposals, not run)

1. `gate_nohole` with seed 43, and `static` with the hole corruption off, to confirm conclusion 4 and separate it from the
   gate (2 runs, ~5 h on two GPUs).
2. A thermal gate that reacts to crossover: a thermal cue measured where depth or RGB sees an object (so warm distractors
   do not count), and a warm start strong enough to hold (weight 1.0 or no annealing), checked early with the training-log
   diagnostics `w_thermal_xo` / `w_thermal_notxo`.
3. Learned attention weighting with a third seed to settle the seed spread, and a look at its missed grasps (approach
   height and gripper timing).
4. Toward 0.8+: grasp precision (more demos or epochs, action horizon), since sensor choice is capped at about 0.75 here.

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
