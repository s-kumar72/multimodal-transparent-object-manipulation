# What does thermal add on top of depth? (plan, written before the runs)

**Goal.** A level where thermal adds to depth. Target success rates on test set (a): rgb at most 0.3-0.4, rgbd
0.5-0.7, rgbdt 0.8-0.9.

**Why these levels.** From `../diagnosis/README.md`:
- L1 with 500 demos reaches 0.84 with rgbdt, so it can be learned. But on L1, depth always shows the glass (the
  probe finds no rgbd vs rgbdt gap), so rgbd ≈ rgbdt is expected there.
- L3b and L4b stayed at 0.2-0.3 with 500 demos, with ImageNet initialization and with milder defects, and most
  failures are missed grasps. Their edge dropout and table hole patches may also break the depth around the gripper
  and hurt grasp precision.

The two new levels therefore start from L2, which can be learned (0.50 with 200 demos), and put the depth failures
on the glass only.

| level | config | what changes from L2 | real sensor it stands in for |
|---|---|---|---|
| L2g | `sim/configs/L2g.yaml` | L3b's blob depth defects (connected holes, blobs of wrong depth), on the glass only | structured-light / stereo depth camera on clear glass |
| L2gs | `sim/configs/L2gs.yaml` | as L2g, but in half of the episodes (`glass_seethrough_p: 0.5`) depth sees through the glass: no holes, no wrong depth, only the surface behind | time-of-flight / LiDAR camera (e.g. RealSense L515) on clear glass |

L2 has no absorbers, table hole patches or edge dropout, so in L2g and L2gs nothing outside the glass changes in
depth. The value 0.5 mixes the two sensor behaviours half and half. It was fixed before the runs and is not changed
after seeing the results.

**Implementation.**
- `SceneConfig.glass_seethrough_p` (default 0.0) in `sim/config.py`.
- `Sensors.episode()` in `sim/sensors.py` draws one extra random number only when the level uses blob defects and
  `glass_seethrough_p > 0`, so existing levels are unchanged.
  - A see-through episode keeps the glass's defect entry but sets missing = 0 and wrong = 0. Without the entry,
    `render()` would fall back to the old dot holes.
- `episode_info["glass_seethrough"]` (0/1) is stored as `meta/ep_glass_seethrough` in the demo sets and in
  `episodes.json`.
- `python policy/summarize_runs.py table` splits success by "depth see-through".

## Checks before the runs

- **Regression.** `sim/regression_test.py --check` gives 190/190 identical arrays for:
  - the default config (`results/data/regress_main.npz`);
  - L3b and L4b, against references saved from the code before the change;
  - L1, L2, L0_faint, L1_alpha02, L3b_mild and L3b_narrow, compared the same way.
- **Depth renders.** `glass_depth_L2g.png` and `glass_depth_L2gs.png`, seeds 0-5, made with
  `python sim/preview_glass_depth.py --levels <level> --seeds 0 1 2 3 4 5 --out <png>`. Each row shows RGB and depth
  at 96×96 as the policy sees them, then the 320×240 depth with the glass outlined in cyan, missing pixels in red and
  wrong-depth pixels in yellow. Per-episode numbers are in the matching `.txt`.
  - In L2gs, seeds 2, 4 and 5 are see-through: no missing or wrong depth on the glass, and the glass is not visible
    in depth. Seeds 0, 1 and 3 have blob defects, the same missing fraction as in L2g.
  - Outside the glass, no pixel is missing in either level.
  - A few cyan dots away from the target are segmentation pixels at blended edges that decode to glass ids. They
    also appear on L0, since `render()` uses the same mask, and cover only a few pixels.
- **See-through share.** 251 of the 500 training seeds (0-499) and 22 of the 50 test seeds (100000-100049) are
  see-through. Test sets (a) and (g) use the same seeds, so they share these 22 episodes.
- **Teacher.** `python sim/teacher_check.py --n 50 --levels L2g L2gs` (with `--workers 16`): 50/50 on training seeds
  and 50/50 on test set (g) for both levels (`teacher_check.txt`).

## Experiment plan

All policy runs: 500 demos (seeds 0-499), 101 epochs, test sets (a) and (g) with 50 episodes each, one seed per run.

1. **Probe**, before any policy training (about 1 h of GPU):
   ```
   python sim/probe.py collect --levels L2g L2gs --workers 32
   python sim/probe.py train --levels L2g L2gs --epochs 25 --seeds 2 --inputs rgb rgbd rgbdt depth thermal \
       --out results/hard_scenario/thermal_value/probe.md
   ```
2. **Smoke test:** `EPOCHS=1 NTEST=2 NENVS=2 GROUP=smoke bash scripts/run_experiment.sh L2gs rgbdt 5` must run to
   the end. Its results, its 5-demo zarr and its `data/runs` folder are then deleted.
3. **Policy runs** (`GROUP=thermal_value`, results in `results/hard_scenario/thermal_value/<run>/`), all six
   whatever the probe shows:
   - L2gs × rgb, rgbd, rgbdt (main candidate, first)
   - L2g × rgb, rgbd, rgbdt
4. **The level × modality table with 500 demos** (`GROUP=level_sweep_500`), in this order and sharing the four GPU
   queues with step 3:
   1. L2 × rgbdt (the baseline for L2g and L2gs)
   2. L1 × rgb, L1 × rgbd
   3. L0 × rgb, L0 × rgbd, L0 × rgbdt
   4. L2 × rgb, L2 × rgbd, L3b × rgb, L3b × rgbd, L4b × rgb

   The 500-demo runs that already exist (L1 rgbdt, L3b rgbdt, L4b rgbd, L4b rgbdt in `../diagnosis/`) are not run
   again. The table goes to `results/hard_scenario/level_sweep_500/README.md`.

## How to read the results

| result | meaning |
|---|---|
| L2gs: rgb ≤ 0.4, rgbd 0.5-0.7, rgbdt ≥ 0.8 | goal reached → a second seed and the L0 control |
| L2gs rgbdt < 0.8 | a learnability problem; compare with L2g to see whether see-through is the cause |
| L2gs rgbd ≈ rgbdt | rgbd does not fail in the see-through episodes (it finds the glass with RGB, for example); check success by condition (see-through 0/1) |
| L2g rgbdt high, rgbd ≈ rgbdt | defects on the glass alone still let depth show the glass |
| rgb > 0.4 | RGB is too strong → consider adding low light (L4's lighting + RGB noise) to the next candidate |

**Expected:** rgbd ≈ 0.5 × rgbdt + 0.5 × rgb, assuming rgbd falls to the rgb level in the see-through half of the
episodes.

With 50 episodes, differences under about 0.15 are within noise
(`python policy/summarize_runs.py compare <run A> <run B>`, Fisher exact test). Whatever the results, all runs are
reported.

## Results

### Probe (2026-11-11, before any policy run)

`probe.md`: the commands above (collection split per level over two GPUs, 16 workers each, same seeds), 25 epochs,
two trainings per input, tested on the first frames of 600 test-set (a) seeds. "Within 3 px / type" is the share of
frames where the target's position is within 3 px / its object type is right. The L2, L3b and L4b rows are from
`../probe.md`.

| level | rgb | rgbd | rgbdt | depth | thermal | rgbd − rgb | rgbdt − rgbd |
|---|---|---|---|---|---|---|---|
| L2 | 0.79 / 0.63 | 0.99 / 0.97 | 0.99 / 0.97 | 0.99 / 0.98 | 0.91 / 0.85 | +0.20 / +0.34 | +0.00 / +0.00 |
| L2g | 0.78 / 0.61 | 0.91 / 0.72 | 0.96 / 0.88 | 0.92 / 0.72 | 0.89 / 0.83 | +0.13 / +0.11 | +0.05 / +0.16 |
| L2gs | 0.79 / 0.62 | 0.82 / 0.64 | 0.92 / 0.84 | 0.72 / 0.47 | 0.90 / 0.84 | +0.03 / +0.02 | +0.10 / +0.20 |
| L3b | 0.76 / 0.56 | 0.91 / 0.73 | 0.94 / 0.83 | 0.90 / 0.70 | 0.89 / 0.81 | +0.15 / +0.17 | +0.03 / +0.10 |
| L4b | 0.73 / 0.54 | 0.90 / 0.72 | 0.94 / 0.84 | 0.90 / 0.69 | 0.89 / 0.81 | +0.17 / +0.18 | +0.04 / +0.12 |

- **On L2gs, depth adds almost nothing to RGB in the probe, and thermal adds the most of any level.** rgbd is 0.03
  / 0.02 above rgb, and rgbdt is 0.10 / 0.20 above rgbd. Depth alone falls to 0.72 / 0.47.
- **L2g puts the thermal gap at L3b's level or above** (+0.05 / +0.16, against +0.03 / +0.10), with depth broken on
  the glass only.
- The probe shows what the first frame carries, not what a policy learns from it. The policy runs below are needed
  for the success rates.
