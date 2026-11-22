# Glass segmentation runs

IoU (%) / loc (% of images whose most confident pixel is on the glass), best checkpoint, clean inputs. Sim columns average the levels the run was tested on; `L4b x` is L4b alone.

| run | train data | modalities | fusion | j_cells11 | k_cells11_heldout | mju_glass | rgbt_glass | transpose | trobi |
|---|---|---|---|---|---|---|---|---|---|
| mix_ws | real:mju_glass, real:rgbd_gsd, real:rgbt_glass, real:transpose, sim | rgbdt | learned/softmax+cues+sup ws | 82.1 / 100.0 | 80.5 / 99.8 | 82.8 / 97.9 | 76.9 / 96.4 | 70.6 / 95.8 | 42.3 / 60.3 |

## Robustness and gates on sim test set a (all levels)

IoU (%) clean and under a corruption applied to every test image; mean gate of the corrupted modality (clean → corrupted). A reliability-aware model should lose little IoU and lower that modality's gate.

| run | clean IoU | thermal crossover IoU | thermal gate | thermal blank IoU | thermal gate | depth holes IoU | depth gate |
|---|---|---|---|---|---|---|---|

## Thermal gate vs. real thermal crossover (sim test set f vs a)

| run | thermal gate, set a (contrast) | thermal gate, set f (crossover) | IoU a | IoU f |
|---|---|---|---|---|
