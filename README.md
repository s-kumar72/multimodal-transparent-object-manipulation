# multimodal-transparent-object-manipulation
Github repo for CMU 11-785 project on multimodal transparent object manipulation.

### MuJoCo Simulation Setup

***Requirements***:
1. Python 3.11 
2. OS: Linux (Ubuntu 22.04/24.04) or macOS 
3. Packages: see sim/requirements.txt

A virtual environment for the simulation can be setup using a Python venv or Conda/ Miniconda using sim/requirements.txt.

### Generating Sim Demo Data

(Run all commands from repo root).

1. Prepare the MJCF of the Kinova Gen3 lite arm
```bash
python sim/prepare_gen3_lite.py
```
2. Check the scene (arm, table, and transparent glass should be visible; selecting 'front_cam' from Rendering should give an angled view of the scene)
```bash
python -m mujoco.viewer --mjcf sim/assets/gen3_lite/transparent_scene.xml
```
3. Check the sensors
```bash
python sim/sensors_test.py
```
Open 'sim/out/sensors.png':
- Top row (glass at 55C): glass is a bright spot in thermal
- Bottom row (thermal crossover): glass disappears from thermal
- Depth: glass is not a solid cylinder

4. Check the teacher using 5 demos
```bash
python sim/collect_demos.py --n 5 --out data/check.zarr
```
Should end with teacher success 5/5 or 4/5 & each demo is 55-70 steps.

5. Check the saved data
```bash
python sim/inspect_dataset.py data/check.zarr
```
No errors & 'sim/out/dataset_preview.png' shows the arm moving to the glass over 5 frames.

6. Generate the full dataset (can generate more than 200 demos if needed)
```bash
python sim/collect_demos.py --n 200 --out data/sim_demos.zarr
python sim/inspect_dataset.py data/sim_demos.zarr
```

### Dataset format

Zarr stores in Diffusion Policy's layout; loads with `ReplayBuffer.copy_from_path(path, keys=[...]). There is one row per timestep at **10 Hz** with all episodes back to back.

| Key | Shape · dtype | Meaning |
|---|---|---|
| `data/rgb` | (N, 96, 96, 3) uint8 | Color image from `front_cam` |
| `data/depth` | (N, 96, 96, 1) uint8 | 0–2 m mapped to 0–255; 0 = no reading (holes on glass) |
| `data/thermal` | (N, 96, 96, 1) uint8 | Simulated thermal, normalized per frame (relative, not °C) |
| `data/joints` | (N, 7) float32 | 6 arm joint angles (rad) + gripper (0 open – 0.85 closed) |
| `data/action` | (N, 7) float32 | Teacher's 6 absolute joint targets (rad) + gripper target |
| `meta/episode_ends` | (episodes,) int64 | Index one past the last step of each episode |

### Thermal Encoder Pretraining (TRansPose)

Pretrains a 1-channel ResNet-18 thermal encoder (GroupNorm, same layout as Diffusion Policy's image encoder) by glass segmentation on TRansPose 14-bit raw thermal. Only the encoder is exported; the FPN segmentation head is dropped.

- `common/thermal.py`: `normalize_thermal_raw(raw, mode, params)` shared by real and sim thermal (`fixed` default, `percentile`, `minmax`).
- `encoder/`: `model.py` (encoder + seg head, BCE + Dice), `dataset.py` (loading, CNR, augmentation), `train.py` (training + evaluation), `export.py` (DP export), `configs/transpose_seg.yaml`.

***Status***: the TRansPose readers in `encoder/dataset.py` (`list_sequences`, `list_frames`, `load_frame`) are stubs until the dataset layout has been inspected, and `THERMAL_RAW_RANGE` is a placeholder (full 14-bit range). Everything else runs on synthetic data (`data.name=random`).

***Requirements***: Python 3.11, `torch`, `torchvision`, `numpy`, `omegaconf`. The DP tests also need the [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) repo at `../diffusion_policy` (or set `DP_ROOT`); otherwise they are skipped.

(Run all commands from repo root. Any config key can be overridden as `key=value`. On a shared login node, set `OMP_NUM_THREADS=4`; otherwise torch starts one thread per core and runs very slowly.)

1. Run the tests (random tensors and synthetic data, CPU)
```bash
OMP_NUM_THREADS=4 python -m unittest common.test_thermal encoder.test_encoder
```
2. Train (`init=imagenet` downloads torchvision's ImageNet ResNet-18 and averages its RGB conv1 into 1 channel)
```bash
python -m encoder.train --config encoder/configs/transpose_seg.yaml data.root=/path/to/TRansPose init=scratch
python -m encoder.train --config encoder/configs/transpose_seg.yaml data.root=/path/to/TRansPose init=imagenet train.lr=3e-4
```
Frames are resized to 96×96 (mask: nearest), cached per sequence in `data.cache_dir`, and split train/val by sequence. Training uses random scaled 84×84 crops, horizontal flip, gain/offset, and noise. `out_dir` (default `data/outputs/encoder/<init>_seed<seed>`) gets `config.yaml`, `splits.json`, `log.csv`, `last.pt`, `best.pt`, and `val_grid.png`.

3. Evaluate a checkpoint on its val split
```bash
python -m encoder.train --eval data/outputs/encoder/scratch_seed0/best.pt
```
The command writes `best_eval.json` and `best_val_grid.png`. The grid has one row per val frame, ordered from low to high CNR, with columns thermal | GT | prediction. Metrics:
- Primary: pixel IoU / F1 on the 84×84 center crop (what DP sees at eval).
- Reference: the same metrics on the full 96×96 frame (`*_full96`).
- Center-crop IoU per thermal-contrast bin (`iou_cnr_low|mid|high`). CNR = |mean glass − mean background| / std background, on raw counts in a ring around the glass. The bin edges are the train-split tertiles (`eval.cnr_edges` overrides them).

4. Export the encoder for Diffusion Policy
```bash
python -m encoder.export data/outputs/encoder/scratch_seed0/best.pt data/outputs/encoder/thermal_resnet18_gn.pth
```
The state_dict keys match DP's `get_resnet("resnet18")` after BN→GN (`fc` excluded). To load the weights:
- `encoder.export.load_thermal_encoder(path)` returns the DP-format module (96×96 1-channel input → 512-d).
- `load_into_dp_obs_encoder(policy.obs_encoder, path, key="thermal")` swaps the weights into a built `MultiImageObsEncoder`. Call it before the optimizer and EMA copy are created.

`--in_chans 3` writes a 3-channel conv1 (weights / 3) for the current DP config, which feeds thermal as 1 channel repeated to 3. **This is a stopgap: DP's ImageNet normalization changes the input scale relative to pretraining. The proper fix is to set thermal to `[1, 96, 96]` in the DP config and turn `imagenet_norm` off for it.**

<!-- 
To setup the virtual environment:
```bash
git clone https://github.com/s-kumar72/multimodal-transparent-object-manipulation.git
cd multimodal-transparent-object-manipulation
```
***Using a Python venv:***
```bash
python3.11 -m venv .venv
source .venv/bin/activate          
pip install --upgrade pip
pip install -r sim/requirements.txt
```

***Using conda/ miniconda:***
```bash
conda create -n transparent-sim python=3.11 -y
conda activate transparent-sim
pip install -r sim/requirements.txt
```
-->
