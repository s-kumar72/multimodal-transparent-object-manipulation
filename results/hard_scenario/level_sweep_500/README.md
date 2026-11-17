# Level × modality with 500 demos

Step 4 of `../thermal_value/README.md`: every level with every input at 500 demos, the data size at which L1 became
learnable (`../diagnosis/`, run 1c). All runs: 500 demos (seeds 0-499), 101 epochs, rollouts on (a) every 25 epochs,
`latest.ckpt` evaluated on test sets (a) in-distribution and (g) held-out with 50 episodes each (seeds
100000-100049), one seed per run.

**Runs.**
- The eleven runs in this folder ran on 2026-11-11 from 07:28 to 18:22 KST, 2.0-3.9 h each including evaluation.
  They shared four RTX A6000s with the six `../thermal_value/` runs and other users' jobs.
- A local runner started each run on whichever GPU had room, with the plan's `scripts/run_experiment.sh` call and
  `MUJOCO_EGL_DEVICE_ID` set to that GPU.
- **One failure, then a retry.** The first attempt at L4b rgb (14:22 KST, GPU 0) ran out of GPU memory in epoch 1,
  after other users' jobs on that GPU grew. The second attempt (15:25 KST, GPU 2) resumed from its `latest.ckpt`
  (epoch 0) with the same settings and finished. No other run failed.
- **Demo sets.** L0 and L2 got new 500-demo sets, collected by the first run of each level. L1, L3b and L4b reuse the
  sets of `../diagnosis/`, and L2g and L2gs those of `../thermal_value/`. The teacher succeeded in 500/500 demos of
  every set.

## Success rate

Test set (a) / (g), 50 episodes each. Each level adds to the row above it, except L2g and L2gs, which start from L2.

| level | scene | rgb | rgbd | rgbdt |
|---|---|---|---|---|
| L0 | one glass on a plain table (`default.yaml`) | 0.96 / 0.94 | 1.00 / 0.98 | 1.00 / 1.00 |
| L1 | 1-3 opaque distractors, textured tables, varied light, liquid, thermal crossover; fainter glass (alpha 0.05-0.1) | 0.02 / 0.02 | 0.82 / 0.84 | 0.84 / 0.78 † |
| L2 | + a look-alike object in every episode | 0.02 / 0.02 | 0.84 / 0.72 | 0.84 / 0.86 |
| L3b | + 1-2 depth absorbers, 1-3 table hole patches, edge dropout, blob depth defects | 0.06 / 0.00 | 0.12 / 0.04 | 0.16 / 0.26 † |
| L4b | + low light and RGB noise | 0.02 / 0.04 | 0.10 / 0.18 † | 0.32 / 0.20 † |
| L2g | L2 + blob depth defects on the glass only | 0.02 / 0.04 ‡ | 0.30 / 0.36 ‡ | 0.40 / 0.36 ‡ |
| L2gs | L2g, but depth sees through the glass in half of the episodes | 0.02 / 0.02 ‡ | 0.14 / 0.26 ‡ | 0.24 / 0.32 ‡ |

**Where each number comes from.** Every run folder is `<level>_<input>_500demos`, under `results/hard_scenario/`:
- unmarked: `level_sweep_500/` (this folder, 11 runs);
- † `diagnosis/` (2026-11-09, same settings):
  - `diagnosis/L1_rgbdt_500demos` (run 1c),
  - `diagnosis/L3b_rgbdt_500demos` (2a),
  - `diagnosis/L4b_rgbd_500demos` (D3),
  - `diagnosis/L4b_rgbdt_500demos` (D2);
- ‡ `thermal_value/` (2026-11-11, same settings, 6 runs).

### Fisher exact tests

p is two-sided, from `python policy/summarize_runs.py compare <run A> <run B>` on (a), and the same test on (g).
With 50 episodes, differences under about 0.15 are within noise.

Between inputs on the same level, p on (a) / (g):

| level | rgb vs rgbd | rgbd vs rgbdt |
|---|---|---|
| L0 | 0.49 / 0.62 | 1 / 1 |
| L1 | 9e-18 / 1e-18 | 1 / 0.61 |
| L2 | 1e-18 / 3e-14 | 1 / 0.14 |
| L3b | 0.49 / 0.49 | 0.77 / **0.004** |
| L4b | 0.20 / 0.051 | **0.013** / 1 |
| L2g | 2e-4 / 9e-5 | 0.40 / 1 |
| L2gs | 0.059 / 9e-4 | 0.31 / 0.66 |

Between levels, success (a) → (a) and (g) → (g), with p:

| step | rgbd | p (a) / (g) | rgbdt | p (a) / (g) |
|---|---|---|---|---|
| L0 → L1 | 1.00 → 0.82, 0.98 → 0.84 | 0.003 / 0.031 | 1.00 → 0.84, 1.00 → 0.78 | 0.006 / 5e-4 |
| L1 → L2 | 0.82 → 0.84, 0.84 → 0.72 | 1 / 0.23 | 0.84 → 0.84, 0.78 → 0.86 | 1 / 0.44 |
| L2 → L3b | 0.84 → 0.12, 0.72 → 0.04 | 2e-13 / 4e-13 | 0.84 → 0.16, 0.86 → 0.26 | 6e-12 / 1e-9 |
| L3b → L4b | 0.12 → 0.10, 0.04 → 0.18 | 1 / 0.051 | 0.16 → 0.32, 0.26 → 0.20 | 0.10 / 0.64 |
| L2 → L2g | 0.84 → 0.30, 0.72 → 0.36 | 7e-8 / 6e-4 | 0.84 → 0.40, 0.86 → 0.36 | 1e-5 / 4e-7 |
| L2g → L3b | 0.30 → 0.12, 0.36 → 0.04 | 0.048 / 9e-5 | 0.40 → 0.16, 0.36 → 0.26 | 0.013 / 0.39 |

For rgb, only L0 → L1 differs (0.96 → 0.02, p = 1e-24 / 2e-23). Every other pair of rgb runs is within 0.06 of each
other (p ≥ 0.49).

This README reports about 70 tests, so three or four p values below 0.05 are expected by chance. The very small p values are far
beyond that. A p near 0.01-0.05 on one test set only (L3b rgbd vs rgbdt, L4b rgbd vs rgbdt, L2g → L3b) needs a second
seed before it counts.

## What the table shows

- **RGB works only on L0.** rgb reaches 0.96 / 0.94 on L0 and 0.00-0.06 on every other level.
  - On L1 and above, rgb runs mostly close the gripper away from every object ("other" in 0.56-0.66 of the episodes
    on (a)), so they do not find the glass.
  - On L0, all three inputs are at 0.94-1.00, with no gap between them (p ≥ 0.24).
- **On L1 and L2, depth carries the task and thermal adds nothing.**
  - rgbd is 0.82 / 0.84 on L1 and 0.84 / 0.72 on L2; rgbdt is 0.84 / 0.78 and 0.84 / 0.86 (rgbd vs rgbdt p ≥ 0.14).
  - The look-alike object of L2 changes nothing for either input (p ≥ 0.23).
- **L0 → L1 costs the depth inputs 0.14-0.22, and the gap is significant** on both test sets (p = 0.003 / 0.031
  for rgbd, 0.006 / 0.0005 for rgbdt).
- **Blob depth defects cause the big drop.**
  - L2 → L3b takes rgbd from 0.84 to 0.12 and rgbdt from 0.84 to 0.16 (p ≤ 1e-9).
  - Blob defects on the glass alone (L2g) already take rgbd to 0.30 / 0.36 and rgbdt to 0.40 / 0.36 (p ≤ 6e-4).
  - The rest of L3b (absorbers, table holes, edge dropout) lowers rgbd further (0.30 → 0.12, p = 0.048 / 9e-5). It
    lowers rgbdt on (a) (0.40 → 0.16, p = 0.013) but not on (g) (0.36 → 0.26, p = 0.39).
  - Low light and RGB noise (L3b → L4b) change neither input significantly (p ≥ 0.051).
  - **With depth in the input, every blob level fails the same way: missed_grasp** in 0.36-0.60 of the episodes,
    against 0.04-0.14 on L1 and L2 (table below).
- **Thermal is ahead of depth only on the blob levels, and never on both test sets.**
  - rgbdt − rgbd is +0.04 / +0.22 on L3b (p = 0.77 / 0.004), +0.22 / +0.02 on L4b (p = 0.013 / 1), +0.10 / 0.00 on
    L2g and +0.10 / +0.06 on L2gs.
  - On these four levels, rgbdt is at or above rgbd in all eight (level, test set) pairs, but rgbdt stays at 0.40
    or below.
- **No level reaches the target of `../thermal_value/`** (rgb ≤ 0.4, rgbd 0.5-0.7, rgbdt ≥ 0.8). Where rgbdt
  reaches 0.8 (L0-L2), rgbd is just as high; where rgbd falls (the blob levels), rgbdt falls with it.
- **500 demos fix L1 and L2 but not the blob levels.** rgbdt on (a) with 200 demos (baselines in
  `../diagnosis/README.md`) → 500 demos:
  - L0 0.92 → 1.00 (p = 0.12);
  - L1 0.60 → 0.84 (p = 0.013);
  - L2 0.50 → 0.84 (p = 0.0006);
  - L3b 0.10 → 0.16 (p = 0.55);
  - L4b 0.14 → 0.32 (p = 0.056).

  On L4b with 200 demos (`../sweep/README.md`), rgb was 0.00 and rgbd 0.08; with 500 demos they are 0.02 and 0.10
  (p = 1 for both).

## Supporting evidence

Per run: success with its 95% Wilson interval, rollouts during training, validation loss, and the two most common
failures.
- missed_grasp: the gripper closed within 8 cm of the target and did not lift it.
- other: the gripper closed away from every object.
- dropped, wrong_height, wrong_object and timeout are in each folder's `episodes.json`. `python
  policy/summarize_runs.py table <run folder>` prints them.

| run | (a) success (95% CI) | (g) success (95% CI) | rollouts on (a), epoch 0 / 25 / 50 / 75 / 100 | lowest val loss (epoch) → at 100 | missed_grasp (a) / (g) | other (a) / (g) |
|---|---|---|---|---|---|---|
| L0 rgb | 0.96 (0.87-0.99) | 0.94 (0.84-0.98) | 0.00 / 0.42 / 0.56 / 0.86 / 0.94 | 0.0013 (91) → 0.0014 | 0.02 / 0.00 | 0.00 / 0.00 |
| L0 rgbd | 1.00 (0.93-1.00) | 0.98 (0.90-1.00) | 0.00 / 0.78 / 0.88 / 0.96 / 1.00 | 0.0016 (91) → 0.0017 | 0.00 / 0.00 | 0.00 / 0.00 |
| L0 rgbdt | 1.00 (0.93-1.00) | 1.00 (0.93-1.00) | 0.00 / 0.74 / 0.94 / 0.94 / 0.98 | 0.0021 (90) → 0.0022 | 0.00 / 0.00 | 0.00 / 0.00 |
| L1 rgb | 0.02 (0.00-0.10) | 0.02 (0.00-0.10) | 0.00 / 0.02 / 0.00 / 0.00 / 0.04 | 0.0024 (96) → 0.0025 | 0.28 / 0.22 | 0.66 / 0.60 |
| L1 rgbd | 0.82 (0.69-0.90) | 0.84 (0.71-0.92) | 0.00 / 0.76 / 0.74 / 0.90 / 0.84 | 0.0017 (99) → 0.0018 | 0.08 / 0.14 | 0.00 / 0.00 |
| L1 rgbdt † | 0.84 (0.71-0.92) | 0.78 (0.65-0.87) | 0.00 / 0.68 / 0.82 / 0.88 / 0.88 | 0.0020 (98) → 0.0023 | 0.04 / 0.08 | 0.00 / 0.02 |
| L2 rgb | 0.02 (0.00-0.10) | 0.02 (0.00-0.10) | 0.00 / 0.02 / 0.02 / 0.04 / 0.06 | 0.0024 (96) → 0.0025 | 0.24 / 0.22 | 0.60 / 0.62 |
| L2 rgbd | 0.84 (0.71-0.92) | 0.72 (0.58-0.83) | 0.00 / 0.70 / 0.76 / 0.80 / 0.86 | 0.0018 (96) → 0.0022 | 0.08 / 0.12 | 0.00 / 0.00 |
| L2 rgbdt | 0.84 (0.71-0.92) | 0.86 (0.74-0.93) | 0.00 / 0.60 / 0.76 / 0.86 / 0.88 | 0.0018 (100) → 0.0018 | 0.06 / 0.10 | 0.00 / 0.00 |
| L3b rgb | 0.06 (0.02-0.16) | 0.00 (0.00-0.07) | 0.00 / 0.00 / 0.02 / 0.00 / 0.04 | 0.0024 (96) → 0.0025 | 0.24 / 0.22 | 0.60 / 0.62 |
| L3b rgbd | 0.12 (0.06-0.24) | 0.04 (0.01-0.13) | 0.00 / 0.06 / 0.10 / 0.12 / 0.06 | 0.0034 (96) → 0.0042 | 0.56 / 0.60 | 0.10 / 0.10 |
| L3b rgbdt † | 0.16 (0.08-0.29) | 0.26 (0.16-0.40) | 0.00 / 0.10 / 0.16 / 0.28 / 0.24 | 0.0053 (66) → 0.0093 | 0.54 / 0.42 | 0.06 / 0.14 |
| L4b rgb | 0.02 (0.00-0.10) | 0.04 (0.01-0.13) | 0.00 / 0.02 / 0.00 / 0.00 / 0.06 | 0.0024 (99) → 0.0024 | 0.18 / 0.24 | 0.66 / 0.66 |
| L4b rgbd † | 0.10 (0.04-0.21) | 0.18 (0.10-0.31) | 0.00 / 0.12 / 0.08 / 0.16 / 0.14 | 0.0035 (86) → 0.0040 | 0.54 / 0.52 | 0.12 / 0.14 |
| L4b rgbdt † | 0.32 (0.21-0.46) | 0.20 (0.11-0.33) | 0.00 / 0.04 / 0.14 / 0.26 / 0.28 | 0.0053 (66) → 0.0088 | 0.46 / 0.52 | 0.10 / 0.06 |
| L2g rgb ‡ | 0.02 (0.00-0.10) | 0.04 (0.01-0.13) | 0.00 / 0.02 / 0.00 / 0.02 / 0.04 | 0.0023 (96) → 0.0025 | 0.30 / 0.18 | 0.56 / 0.66 |
| L2g rgbd ‡ | 0.30 (0.19-0.44) | 0.36 (0.24-0.50) | 0.00 / 0.30 / 0.34 / 0.44 / 0.38 | 0.0024 (96) → 0.0025 | 0.46 / 0.38 | 0.02 / 0.02 |
| L2g rgbdt ‡ | 0.40 (0.28-0.54) | 0.36 (0.24-0.50) | 0.00 / 0.22 / 0.32 / 0.38 / 0.38 | 0.0034 (87) → 0.0040 | 0.38 / 0.38 | 0.02 / 0.00 |
| L2gs rgb ‡ | 0.02 (0.00-0.10) | 0.02 (0.00-0.10) | 0.00 / 0.02 / 0.00 / 0.02 / 0.02 | 0.0023 (96) → 0.0024 | 0.26 / 0.22 | 0.62 / 0.62 |
| L2gs rgbd ‡ | 0.14 (0.07-0.26) | 0.26 (0.16-0.40) | 0.00 / 0.10 / 0.14 / 0.08 / 0.10 | 0.0029 (87) → 0.0030 | 0.52 / 0.36 | 0.12 / 0.24 |
| L2gs rgbdt ‡ | 0.24 (0.14-0.37) | 0.32 (0.21-0.46) | 0.00 / 0.18 / 0.16 / 0.24 / 0.34 | 0.0046 (66) → 0.0070 | 0.48 / 0.48 | 0.06 / 0.06 |

- **An earlier checkpoint would not change the picture.** On the blob levels, no rollout goes above 0.44 (L2g rgbd
  at epoch 75), and rgb on L1 and above never goes above 0.06.
- **rgbdt's validation loss rises late on three of the four blob levels.** It goes from its lowest point at epoch
  66 to 0.0070-0.0093 at epoch 100 on L3b, L4b and L2gs.
  - On L2g it rises less (0.0034 → 0.0040), and rgbd on the blob levels rises at most from 0.0034 to 0.0042.
  - Every other run stays flat at 0.0013-0.0025.
  - This is only an observation; the evaluations use `latest.ckpt` as planned.
