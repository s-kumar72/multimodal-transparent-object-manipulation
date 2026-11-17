# Glass segmentation runs

IoU (%) / loc (% of images whose most confident pixel is on the glass), best checkpoint, clean inputs. Sim columns average the levels the run was tested on; `L4b x` is L4b alone.

| run | train data | modalities | fusion | sim a (in-dist.) | sim f (crossover) | sim g (held out) | mju_glass | rgbt_glass | transpose | trobi |
|---|---|---|---|---|---|---|---|---|---|---|
| mix_ws | real:mju_glass, real:rgbd_gsd, real:rgbt_glass, real:transpose, sim | rgbdt | learned/softmax+cues+sup ws | 87.1 / 100.0 | 84.0 / 100.0 | 86.0 / 99.3 | 82.7 / 97.9 | 76.7 / 96.9 | 70.3 / 93.7 | 44.3 / 53.3 |

## Robustness and gates on sim test set a (all levels)

IoU (%) clean and under a corruption applied to every test image; mean gate of the corrupted modality (clean → corrupted). A reliability-aware model should lose little IoU and lower that modality's gate.

| run | clean IoU | thermal crossover IoU | thermal gate | thermal blank IoU | thermal gate | depth holes IoU | depth gate |
|---|---|---|---|---|---|---|---|
| mix_ws | 87.1 | 85.9 | 0.29 → 0.00 | 84.5 | 0.29 → 0.00 | 85.1 | 0.35 → 0.31 |

## Thermal gate vs. real thermal crossover (sim test set f vs a)

| run | thermal gate, set a (contrast) | thermal gate, set f (crossover) | IoU a | IoU f |
|---|---|---|---|---|
| mix_ws | 0.29 | 0.05 | 87.1 | 84.0 |
