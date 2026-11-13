# Why does RGBDT fall on the hard levels? (plan, written before the runs)

**Where we are.** With 200 demos and 101 epochs, RGBDT reached 0.92 on L0, 0.60 on L1, 0.50 on L2, 0.10 on L3b and
0.14 on L4b (test set (a), 50 episodes); RGB on L4b reached 0.00. L1 with 200 demos for 201 epochs stayed at 0.56.
**Goal:** a level where RGB is low (≤ 0.3-0.4 is fine) and RGBDT reaches 0.8-0.9.

**Runs.** One change from the 200-demo runs each; everything else as in `../sweep/` (101 epochs, rollouts on (a)
every 25 epochs, the last checkpoint on test sets (a) and (g), 50 episodes each). Variant rgbdt unless noted.
"ImageNet" = `policy.obs_encoder.rgb_model.weights=IMAGENET1K_V1`: every image encoder (rgb, depth, thermal) starts
from ImageNet ResNet-18 weights instead of random ones; the architecture is unchanged.

| id | level | demos | change | question | result folder |
|---|---|---|---|---|---|
| D2 | L4b | 500 | more demos | is the main scenario a data problem? | `L4b_rgbdt_500demos` |
| D3 | L4b (rgbd) | 500 | more demos | with more data, does thermal add over rgbd? | `L4b_rgbd_500demos` |
| P4 | L4b | 200 | ImageNet | is it the randomly initialized encoders? | `L4b_rgbdt_200demos_imagenet` |
| 1a | L0_faint | 200 | L0 + only the faint glass | does the faint glass alone cause the L0 -> L1 drop? | `L0_faint_rgbdt_200demos` |
| 1b | L1_alpha02 | 200 | L1 without the faint glass | does L1 recover without it? | `L1_alpha02_rgbdt_200demos` |
| 1c | L1 | 500 | more demos | is the L1 drop a data problem? | `L1_rgbdt_500demos` |
| 1d | L1 | 200 | ImageNet | is the L1 drop an encoder problem? | `L1_rgbdt_200demos_imagenet` |
| 2a | L3b | 500 | more demos | is the L2 -> L3b drop a data problem? | `L3b_rgbdt_500demos` |
| 2a' | L3b | 200 | ImageNet | ... an encoder problem? | `L3b_rgbdt_200demos_imagenet` |
| 2b | L3b_mild | 200 | depth defects: missing 0.1-0.5, wrong depth 0-10% | is it how strong the defects are? | `L3b_mild_rgbdt_200demos` |

The new levels are `sim/configs/L0_faint.yaml`, `L1_alpha02.yaml` and `L3b_mild.yaml`. 500 demos x 101 epochs is
about 41k steps, the budget at which the 500-demo main-scene run reached 0.96; L1 with 200 demos for 201 epochs
already had about 34k steps, so a gain at 500 demos comes from the data. 500-demo sets use seeds 0-499 and contain
the 200-demo sets.

**Baselines (200 demos, rgbdt, test set (a)):** L0 0.92, L1 0.60, L2 0.50, L3b 0.10, L4b 0.14 (`../sweep/L4b_rgbdt`);
L4b rgbd 0.08 (`../sweep/L4b_rgbd`).

**Reading the results.** With 50 episodes, differences under about 0.15 are within noise; use
`python policy/summarize_runs.py compare <baseline run> <new run>` (Fisher exact test).

| finding | meaning |
|---|---|
| 1a low, 1b near 0.9 | the faint glass causes the L1 drop |
| 1a near 0.9, 1b low | the rest of the shared scene (distractors, tables, light, liquid, thermal) causes it |
| 1c or 1d well above 0.60 | the L1 drop is (also) data / encoder |
| 2a or 2a' well above 0.10 | the L3b drop is data / encoder; 2b high: the defects are too strong |
| D2 or P4 near 0.8 | L4b becomes usable with that change; then rgb / rgbd / rgbdt and L0 with it |

Whatever the results, all ten runs are reported.

**Run:** `bash scripts/run_diagnosis.sh` (four GPUs; `scripts/run_experiment.sh` for a single run). Results land in
`results/hard_scenario/diagnosis/<result folder>/`.

## Results

**All ten runs finished and none failed** (no `!!! FAILED` in the runner or queue logs). They ran on 2026-11-09 from
12:48 to 19:14 KST on four RTX A6000s shared with other users' jobs: 2.5-3.3 h per 500-demo run and 1.4-1.8 h per
200-demo run, including evaluation. The settings are the plan's: 200 or 500 demos, 101 epochs, rollouts on (a) every
25 epochs, the last checkpoint on (a) and (g) with 50 episodes each, one seed per run. The teacher succeeded in every
demo of the new sets (500/500 on L1, L3b and L4b; 200/200 on L0_faint, L1_alpha02 and L3b_mild). The 200-demo runs on
L1, L3b and L4b reused the sweep's demo sets. Instead of the fixed queue per GPU of `scripts/run_diagnosis.sh`, a
local runner started each run on whichever GPU had room, with the same `scripts/run_experiment.sh` call and
`MUJOCO_EGL_DEVICE_ID` set to that GPU.

**Baselines.** L4b rgbdt and rgbd are `../sweep/L4b_rgbdt` and `../sweep/L4b_rgbd`. L0, L1 and L3b are the 200-demo
sweep runs of 2026-11-08 (`L0_rgbdt`, `L1_rgbdt`, `L3b_rgbdt`). Their folders are not in git, but `compare` read them,
and their counts on (a) are the plan's numbers (46, 30 and 5 of 50).

### Success rate

Difference and p are on test set (a), against the 200-demo baseline. p is the two-sided Fisher exact test from
`python policy/summarize_runs.py compare <baseline> <run>`. With 50 episodes, differences under about 0.15 are
within noise.

| id | level | demos | change | (a) success (95% CI) | (g) success | baseline (a) | difference | p |
|---|---|---|---|---|---|---|---|---|
| D2 | L4b | 500 | more demos | 0.32 (0.21-0.46) | 0.20 | 0.14 (L4b) | +0.18 | 0.056 |
| D3 | L4b, rgbd | 500 | more demos | 0.10 (0.04-0.21) | 0.18 | 0.08 (L4b rgbd) | +0.02 | 1.0 |
| P4 | L4b | 200 | ImageNet | 0.20 (0.11-0.33) | 0.10 | 0.14 (L4b) | +0.06 | 0.60 |
| 1a | L0_faint | 200 | L0 + faint glass | 0.98 (0.90-1.00) | 0.96 | 0.92 (L0) | +0.06 | 0.36 |
| 1b | L1_alpha02 | 200 | L1 without faint glass | 0.68 (0.54-0.79) | 0.56 | 0.60 (L1) | +0.08 | 0.53 |
| 1c | L1 | 500 | more demos | 0.84 (0.71-0.92) | 0.78 | 0.60 (L1) | +0.24 | 0.014 |
| 1d | L1 | 200 | ImageNet | 0.66 (0.52-0.78) | 0.56 | 0.60 (L1) | +0.06 | 0.68 |
| 2a | L3b | 500 | more demos | 0.16 (0.08-0.29) | 0.26 | 0.10 (L3b) | +0.06 | 0.55 |
| 2a' | L3b | 200 | ImageNet | 0.24 (0.14-0.37) | 0.12 | 0.10 (L3b) | +0.14 | 0.11 |
| 2b | L3b_mild | 200 | milder depth defects | 0.14 (0.07-0.26) | 0.20 | 0.10 (L3b) | +0.04 | 0.76 |
| D2 vs D3 | L4b | 500 | rgbdt instead of rgbd | 0.32 vs 0.10 | 0.20 vs 0.18 | 0.10 (D3) | +0.22 | 0.013 |

- **Only one change gives a clear gain: 500 demos on L1 (1c), 0.60 → 0.84 (p = 0.014).** D2 gains 0.18, but
  p = 0.056. Every other run is within 0.15 of its baseline on (a).
- **(g) mostly agrees, with one exception.** 2a rises on (g) from 0.08 to 0.26 (p = 0.031) but not on (a) (p = 0.55).
  1c goes from 0.64 to 0.78 on (g) (p = 0.19) and D2 from 0.10 to 0.20 (p = 0.26). The other runs are within 0.12 of
  their baseline on (g) (p ≥ 0.14).
- **These are 22 tests** (11 rows × two test sets), so about one p below 0.05 is expected by chance. Numbers close
  to the line, such as 2a on (g) or D2 on (a), need a second seed before they count.

### What causes the drops

**L0 → L1 comes from the rest of the L1 scene, and 500 demos largely fix it.** The faint glass and the encoder
initialization are not the cause.
- **The faint glass alone costs nothing.** L0 with it (1a) reached 0.98, against 0.92 (p = 0.36); (g) is 0.96 for
  both.
- **L1 without the faint glass (1b) stays at 0.68** (p = 0.53 against 0.60). 1a vs 1b is 0.98 vs 0.68 (p = 8e-5).
  In the interpretation table this is "1a near 0.9, 1b low": the shared scene (distractors, tables, light, liquid,
  thermal) causes the drop.
- **500 demos (1c) bring L1 to 0.84.** That is within noise of L0 on (a) (0.92, p = 0.36), but still below L0 on
  (g) (0.78 vs 0.96, p = 0.015).
- **ImageNet initialization (1d) only speeds up the start.** It is at 0.58 at epoch 25 against 0.30, then levels
  off at 0.72-0.74 in the rollouts and ends at 0.66 (p = 0.68).
- Of the three candidates, data (1c) has by far the largest effect.

**None of the three changes brings L3b back.** By the 0.15 rule, none of 2a, 2a' or 2b is "well above 0.10".
- **ImageNet (2a') comes closest:** 0.24 (+0.14, p = 0.11), but 0.12 on (g).
- **500 demos (2a):** 0.16 on (a) (p = 0.55) and 0.26 on (g) (p = 0.031).
- **Milder defects (2b; missing 0.1-0.5, wrong depth 0-10%):** 0.14 (p = 0.76), still far below L2's 0.50
  (p = 0.0002).
- **2b cannot isolate the cause.** It changes only the strength of the blob defects. L3b also keeps L3's
  absorbers, table holes and edge dropout, so this run cannot say whether those, or blob defects of any strength,
  cause the drop.

**L4b gets nowhere near 0.8 with 500 demos or with ImageNet.** D2 reached 0.32 (95% CI 0.21-0.46, p = 0.056) and P4
reached 0.20 (0.11-0.33, p = 0.60). Even the top ends of both intervals are far below 0.8, so these changes do not
make L4b usable.

**With 500 demos, rgbdt is ahead of rgbd on (a) but not on (g).**
- **On (a), D2 (rgbdt) scores 0.32 and D3 (rgbd) 0.10 (p = 0.013).** At 200 demos they were 0.14 and 0.08
  (p = 0.53). rgbd gained nothing from 500 demos (0.08 → 0.10, p = 1.0), while rgbdt gained 0.18.
- **The gap is in the episodes where thermal can show the glass.** Without thermal crossover, rgbdt succeeds in
  16/44 and rgbd in 3/44. In the 6 crossover episodes, rgbdt succeeds in 0 and rgbd in 2.
- **On (g) the two are equal** (0.20 vs 0.18, p = 1.0). Each is a single seed, so this is a lead to check, not a
  result.

### Supporting evidence

Rollouts during training, validation loss and failures on test set (a). wrong_object and timeout are 0.00-0.02 in
every run and are left out.

| run | rollouts on (a), epoch 0 / 25 / 50 / 75 / 100 | lowest val loss (epoch) → at 100 | missed_grasp | dropped | wrong_height | other |
|---|---|---|---|---|---|---|
| L0, 200 | 0.00 / 0.42 / 0.64 / 0.84 / 0.96 | 0.0036 (95) → 0.0056 | 0.04 | 0.04 | 0.00 | 0.00 |
| 1a L0_faint | 0.00 / 0.34 / 0.74 / 0.84 / 0.96 | 0.0036 (92) → 0.0042 | 0.02 | 0.00 | 0.00 | 0.00 |
| L1, 200 | 0.00 / 0.30 / 0.40 / 0.50 / 0.62 | 0.0044 (98) → 0.0051 | 0.08 | 0.18 | 0.14 | 0.00 |
| 1b L1_alpha02 | 0.00 / 0.34 / 0.42 / 0.54 / 0.56 | 0.0044 (97) → 0.0049 | 0.08 | 0.06 | 0.16 | 0.02 |
| 1c L1, 500 | 0.00 / 0.68 / 0.82 / 0.88 / 0.88 | 0.0020 (98) → 0.0023 | 0.04 | 0.10 | 0.02 | 0.00 |
| 1d L1, ImageNet | 0.00 / 0.58 / 0.74 / 0.72 / 0.74 | 0.0081 (66) → 0.0096 | 0.06 | 0.16 | 0.12 | 0.00 |
| L3b, 200 | 0.00 / 0.02 / 0.14 / 0.10 / 0.08 | 0.0115 (56) → 0.0217 | 0.34 | 0.22 | 0.18 | 0.14 |
| 2a L3b, 500 | 0.00 / 0.10 / 0.16 / 0.28 / 0.24 | 0.0053 (66) → 0.0093 | 0.54 | 0.14 | 0.10 | 0.06 |
| 2a' L3b, ImageNet | 0.00 / 0.06 / 0.26 / 0.22 / 0.20 | 0.0173 (45) → 0.0427 | 0.34 | 0.12 | 0.08 | 0.22 |
| 2b L3b_mild | 0.02 / 0.04 / 0.18 / 0.12 / 0.18 | 0.0119 (45) → 0.0230 | 0.38 | 0.18 | 0.10 | 0.18 |
| L4b, 200 | 0.02 / 0.02 / 0.12 / 0.18 / 0.16 | 0.0123 (45) → 0.0239 | 0.52 | 0.12 | 0.06 | 0.16 |
| D2 L4b, 500 | 0.00 / 0.04 / 0.14 / 0.26 / 0.28 | 0.0053 (66) → 0.0088 | 0.46 | 0.08 | 0.04 | 0.10 |
| P4 L4b, ImageNet | 0.00 / 0.04 / 0.22 / 0.12 / 0.22 | 0.0160 (45) → 0.0392 | 0.56 | 0.16 | 0.00 | 0.06 |
| L4b rgbd, 200 | 0.00 / 0.02 / 0.00 / 0.06 / 0.12 | 0.0096 (67) → 0.0157 | 0.58 | 0.06 | 0.06 | 0.18 |
| D3 L4b rgbd, 500 | 0.00 / 0.12 / 0.08 / 0.16 / 0.14 | 0.0035 (86) → 0.0040 | 0.54 | 0.06 | 0.16 | 0.12 |

- **Overfitting depends on the level, not the change.**
  - On L0 and L1, validation loss is lowest at epoch 92-98 (1d: 66). The new runs end within 1.2× of their minimum
    (the L0 baseline at 1.6×).
  - On L3b and L4b it is lowest at epoch 45-66 and then rises to 1.7× with 500 demos (D2, 2a), 1.9× for 2b and 2.5×
    with ImageNet (P4, 2a').
  - 500 demos give less than half the minimum (0.0053 against 0.0115-0.0123), but success barely moves.
  - rgbd with 500 demos (D3) does not overfit (minimum at epoch 86) and does not improve either.
  - Each run validates on episodes from its own demo set, so losses compare best within one level and demo count.
- **ImageNet speeds up early learning but does not change the end result.** At epoch 25 or 50, the ImageNet runs
  are ahead of their baselines: 1d 0.58 vs 0.30 at epoch 25, 2a' 0.26 vs 0.14 and P4 0.22 vs 0.12 at epoch 50.
  Their final numbers are within noise of the baseline, and their validation loss is the highest on each level.
- **Still rising at epoch 100:** 1a (0.84 → 0.96). 1c levels off at 0.88 from epoch 75, and D2 is almost flat
  (0.26 → 0.28).
- **On L1, 500 demos remove most of the drops and wrong-height grasps** (0.18 → 0.10 and 0.14 → 0.02). 1b and 1d
  keep the baseline's wrong-height grasps (0.16 and 0.12, against 0.14).
- **On L4b, missed grasps stay the main failure in every run** (0.46-0.56, against 0.52 at 200 demos). What changes
  is "other", closing away from every object: from 0.16 to 0.10 (D2) and 0.06 (P4). The policy finds the target more
  often but still does not grasp it.
- **On L3b, 500 demos move failures into missed grasps** (0.34 → 0.54).
- **Weak conditions on (a):**
  - The skyblue table was 0/14 on L4b with 200 demos. It is 4/14 with D2, 1/14 with P4, and 2/14 for rgbd with 500
    demos (D3; 1/14 at 200).
  - Thermal crossover episodes: 0/6 → 0/6 (D2), 2/6 (P4), 2/6 (D3).
  - D2 is weakest on the flask (0/10, 2/10 at 200 demos), spot light (2/16) and the noise table (2/10). P4 is
    weakest on the flask (0/10), two distractors (0/19) and the solid table (0/10).
  - On L3b, dim light is 0/14 in the baseline, 2a and 2a' (2/14 in 2b). The tumbler is 0/10 in the baseline and 2a'.
  - On L1, the skyblue table is the weakest table at baseline (4/14). It is 8/14 in 1b, 6/14 in 1d and 12/14 in 1c.
    The flask is the weakest object (2/10 at baseline, 6/10 in 1c).

### Next step (proposal, not run)

**Level: L1 with 500 demos.** It is the only setting of the ten where rgbdt reaches 0.8-0.9: 0.84 on (a) (95% CI
0.71-0.92) and 0.78 on (g). Whether RGB is low on L1 has not been measured. Two things suggest it is:
- In the probe (`../probe.md`), rgb places the target within 3 px in 0.82 of L1 frames and gets the object type
  right in 0.66, against 0.99 and 0.97 for rgbdt.
- L1's faint glass is meant to weaken RGB, and 1a shows that it does not hurt rgbdt.

Formal runs: 500 demos, 101 epochs, (a) and (g) with 50 episodes each.

| level | rgb | rgbd | rgbdt |
|---|---|---|---|
| L1 | new | new | 1c (0.84), plus a second seed |
| L0 (control) | new | new | new |

- **Order and cost.** Run L1 × rgb first: if it reaches about 0.8, L1 is not the level, and the other runs should
  wait. The L0 500-demo set still has to be collected (`scripts/run_experiment.sh` does it). The six runs take about
  2.5-3 h each on these shared A6000s, about 6 h on four GPUs.
- **Thermal is unlikely to matter on L1.** The probe shows no rgbd vs rgbdt gap there (+0.00 within 3 px, -0.01 on
  object type); it is clear only on L3b and L4b (+0.10 and +0.12 on object type). Expect rgb < rgbd ≈ rgbdt. The only sign of rgbdt > rgbd so far is
  D2 vs D3 on L4b (0.32 vs 0.10 on (a), p = 0.013, not on (g)), at success rates far below 0.8. If the thermal gap
  is needed, a second seed of D2 and D3 is the cheapest check.
- **Cheaper alternative: L0_faint with 200 demos.** rgbdt reached 0.98 there (1a). If L0_faint × rgb (one run of
  about 1.5 h) is low, L0_faint isolates transparency with 200 demos. There is no probe for L0_faint yet.
