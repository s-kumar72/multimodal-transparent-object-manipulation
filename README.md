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
2. Check the scene (arm, table, and a transparent tumbler should be visible; selecting 'front_cam' from Rendering should give an angled view of the scene). The other four objects are parked on the floor behind the camera; `env.py` moves one of them onto the table per episode.
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

4. Check the objects
```bash
python sim/objects_test.py
```
Prints, per object, the pixels it covers in the camera, its height, and the teacher's success on 3 seeds (should be 3/3 for every object), and checks that the parked objects are out of view. Open 'sim/out/objects.png': one row per object (RGB, depth, thermal).

5. Check the teacher using 5 demos
```bash
python sim/collect_demos.py --n 5 --out data/check.zarr
```
Should end with teacher success 5/5 for every object that was drawn & each demo is 56-70 steps. `--obj bottle` collects one object only.

6. Check the saved data
```bash
python sim/inspect_dataset.py data/check.zarr
```
No errors, the episodes-per-object line lists the objects, & 'sim/out/dataset_preview.png' shows the arm moving to the object over 5 frames.

7. Generate the full dataset (can generate more than 200 demos if needed; the object is drawn uniformly at random per episode)
```bash
python sim/collect_demos.py --n 200 --out data/sim_demos.zarr
python sim/inspect_dataset.py data/sim_demos.zarr
```

### Transparent objects

One object per episode, drawn uniformly at random, standing upright at a random spot on the table (x 0.30-0.42 m, y -0.15-0.15 m in front of the arm). All are glass-like primitives in `sim/assets/gen3_lite/transparent_scene.xml`; the scripted teacher grasps them top-down at the height in `OBJECTS` in `sim/env.py` and lifts 16 cm (less for tall objects, the top-down IK cannot reach above 0.30 m at the far edge of the table). Success is a 10 cm lift.

| Object | Shape | Size | Teacher grasps |
|---|---|---|---|
| `tumbler` | cylinder | 6 cm wide, 10 cm tall | middle of the cylinder |
| `bottle` | cylinder + neck with a lip | 6 cm wide, 17 cm tall (2.6 cm neck) | the neck, under the lip |
| `wineglass` | base, stem, ellipsoid bowl | 6 cm wide, 16 cm tall | the bowl, just below its widest point |
| `jar` | box | 5 x 5 x 8 cm | middle of the box |
| `flask` | base, sphere, neck with a lip | 7 cm wide, 15 cm tall (2.6 cm neck) | the neck, under the lip |

Necks and the bowl are grasped right under something wider so the object cannot slip out of the fingers. Transparent geoms are in geom group 5, which is how `sensors.py` tells them apart for the depth holes.

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
| `meta/object_id` | (episodes,) int64 | Which object the episode used; names are in the `object_names` attribute of `meta` |

### Training Diffusion Policy on the sim data

Uses the original [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) code (CNN U-Net, ResNet18 image encoders) with our own dataset, eval env, and config in `policy/`.
Depth and thermal are repeated to 3 channels so each modality gets its own ResNet18 encoder; joint angles are always included.

**On PSC Bridges-2, follow `scripts/bridges2/README.md`**: one setup script, then sbatch jobs to train or to evaluate a
checkpoint. Results of the RGB run on the five objects are in `results/rgb_5objects/README.md`. The steps below are the
same thing on any Linux machine with an NVIDIA GPU.

1. Clone Diffusion Policy next to this repo and install its dependencies into the sim environment (Python 3.11).
   gym 0.21 only builds with an old pip, setuptools and wheel, and diffusers 0.11.1 needs `huggingface_hub<0.26`.
   `scripts/bridges2/setup.sh` has the full, tested list.
```bash
git clone https://github.com/real-stanford/diffusion_policy.git ../diffusion_policy
pip install "pip<24.1" "setuptools<66" "wheel<0.40"
pip install --no-build-isolation gym==0.21.0
pip install diffusers==0.11.1 "huggingface_hub<0.26" hydra-core einops dill numba av imagecodecs wandb pandas pymunk shapely scikit-image imageio-ffmpeg
pip install -e ../diffusion_policy --no-deps
```
2. Generate demos, or download the 500 five-object demos (`sim_demos_5obj.zarr.zip`) from the release `multi-object-v0.1`.
   `--workers` runs episodes in parallel; rendering is the bottleneck. Workers are forked on Linux and spawned on macOS,
   where a forked process cannot render.
```bash
python sim/collect_demos.py --n 500 --workers 16 --out data/sim_demos.zarr
```
3. Train one modality combination (`rgb`, `rgbd`, `rgbt`, or `rgbdt`). Every 25 epochs the policy is rolled out in unseen
   test scenes; `test/mean_score` is the success rate and `test/mean_score_<object>` the rate per object. Diffusion Policy
   evaluates and saves checkpoints only when `epoch % 25 == 0` (epochs count from 0), so use 151 epochs for 150 epochs
   plus a final evaluation.
```bash
CUDA_VISIBLE_DEVICES=0 policy/train.sh rgbdt data/sim_demos.zarr training.num_epochs=151 task.env_runner.n_test=50
```
On a headless Linux server set `MUJOCO_GL=egl` (or `osmesa`). Logs, checkpoints, and rollout videos go to `data/outputs/`.

4. Evaluate a checkpoint, e.g. the RGB policy from the release (`rgb_5obj_epoch100_ema.ckpt`). Test scene k uses seed
   `test_start_seed + k`; other seeds give new scenes drawn the same way.
```bash
PYTHONPATH=$PWD:$PWD/sim:../diffusion_policy MUJOCO_GL=egl \
    python policy/eval.py -c data/checkpoints/rgb_5obj_epoch100_ema.ckpt -o data/eval/rgb --n_test 50 --test_start_seed 100000
```

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
