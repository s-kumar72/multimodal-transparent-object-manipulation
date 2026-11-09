# Is 200 demos too few? (plan, written before the runs)

**Question.** RGBDT fell from 0.92 (L0) to 0.60 (L1) and 0.14 (L4b) with 200 demos and 101 epochs. Is that the data
budget or the levels? The main-scene RGB result that reached 0.96 used 500 demos.

**Runs** (only the number of demos changes; everything else as in `../sweep/`):

| run | level | input | demos | epochs | compare with |
|---|---|---|---|---|---|
| D2 | L4b | rgbdt | 500 | 101 | L4b rgbdt, 200 demos: 0.14 on (a) (`../sweep/L4b_rgbdt`) |
| D1 | L1 | rgbdt | 500 | 101 | L1 rgbdt, 200 demos: 0.60 at 101 epochs, 0.56 at 201 epochs |

D2 first. 500 demos x 101 epochs = about 41k steps, the budget at which the 500-demo main-scene run reached 0.96
(epoch 100). L1 with 200 demos for 201 epochs already had about 34k steps and stayed at 0.56, so a gain at 500 demos
would come from the data, not from more steps. The 500-demo sets use seeds 0-499, so they contain the 200-demo sets.

**Reading the results** (50 episodes per test set: a difference under about 0.15 is within noise; use
`python policy/summarize_runs.py compare` for a Fisher exact test):

| outcome | meaning | next |
|---|---|---|
| D2 near 0.8 on (a) | the data budget was the problem | rgb / rgbd / rgbdt and L0 with 500 demos |
| D1 up a lot, D2 still low | L1's drop was data; L3b's depth defects are a separate problem | milder L3b defects |
| both about the same | not the data | glass alpha back to 0.2 on L1, ImageNet-pretrained encoders |

**Run:** `bash scripts/run_experiment.sh L4b rgbdt 500`, then `bash scripts/run_experiment.sh L1 rgbdt 500`.
Results land in `results/hard_scenario/demo_scale/<level>_rgbdt_500demos/`.
