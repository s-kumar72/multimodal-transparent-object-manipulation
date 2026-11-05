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
| `data/thermal` | (N, 96, 96, 1) uint8 | Simulated thermal, normalized as the scene config says (default: per-frame min-max, relative, not °C) |
| `data/thermal_raw` | (N, 96, 96, 1) uint16 | Simulated 14-bit thermal counts (gain · band radiance + offset); not in zarrs made before the scene configs |
| `data/joints` | (N, 7) float32 | 6 arm joint angles (rad) + gripper (0 open – 0.85 closed) |
| `data/action` | (N, 7) float32 | Teacher's 6 absolute joint targets (rad) + gripper target |
| `meta/episode_ends` | (episodes,) int64 | Index one past the last step of each episode |
| `meta/object_id` | (episodes,) int64 | Which object the episode used; names are in the `object_names` attribute of `meta` |
| `meta/ep_*` | (episodes, ...) numbers | Scene conditions of each episode (seed, split, level, background, lighting, distractors, liquid, temperatures, crossover, ...); names of categorical ids in `meta` attributes, the whole config as JSON in `meta.attrs["scene_config"]` |

Images are stored 64 frames per zarr chunk.

### Scene configs and the hard scenario

`sim/env.py` takes a scene config (`GlassPickEnv(config="L3.yaml")`, `collect_demos.py --config sim/configs/L3.yaml`).
Configs are YAML files in `sim/configs/` that can inherit (`base: other.yaml`); `sim/config.py` lists every key with
its default. **The default config is main's scene**: `python sim/regression_test.py --check <npz>` checks that depth,
thermal, joints and actions are bit-identical to a reference recorded on main (RGB may differ by 1 on a few pixels,
EGL's alpha blending is not reproducible run to run). New random factors draw from their own RNG stream
(`default_rng([seed, factor])`), so the same seed gives the same object, position and temperatures at every level.

**Difficulty levels** (fixed before any policy was trained; each adds to the previous one):

| Level | Config | Adds | Real-world failure it stands for |
|---|---|---|---|
| L0 | `L0.yaml` | main's scene | - |
| L1 | `L1.yaml` | shared hard scene (`hard_base.yaml`) + **lever A** | Glass is hard to see in colour images. MuJoCo draws glass by alpha blending only (no refraction, caustics or Fresnel edges), so rendered glass at alpha 0.2 is easier to see than real glass: alpha 0.05-0.1, a slight tint, and glass-like sky-blue / white tables and dim / back light stand in for that. |
| L2 | `L2.yaml` | **lever B**: an opaque look-alike of the target | Opaque cups and bottles of the same shape (ceramic, plastic) are common; in RGB they are the clearer object, while depth and thermal see a solid object. |
| L3 | `L3.yaml` | **lever C**: depth absorbers, table hole patches, edge dropout | Active depth cameras return no depth on IR-absorbing matte black and on specular metal, drop pixels on glossy surfaces and in sunlight (IR interference), and lose pixels at depth edges ("flying pixels"). So a patch of missing depth is no longer only glass. In thermal these differ from glass: bare metal (emissivity 0.05-0.3) reflects the ambient, black paint (0.95-0.98) shows its own temperature. |
| L4 | `L4.yaml` | low light: dim / back / spot light only, Gaussian noise (std 6) on the RGB image | A dark room or a window behind the table: underexposed images and shot / read noise. Active-IR depth and thermal do not depend on visible light. |

| L3b | `L3b.yaml` | L3 with **structured depth defects** (`depth_defects: blobs`) | see below |
| L4b | `L4b.yaml` | L4 with the depth defects of L3b | see below |
| L3b_narrow, L4b_narrow | `L3b_narrow.yaml`, `L4b_narrow.yaml` | L3b / L4b seen by the narrow camera (target ~2.3% of the image instead of ~0.26%) | - |

**Structured depth defects (L3b, L4b).** In L3 every glass pixel is missing independently with probability 0.4, and
that dot pattern still outlines the glass. With `depth_defects: blobs`, the glass, the absorbers and the table hole
patches get the same defect model, drawn per region and episode from the same ranges (`sim/sensors.py`,
`Sensors._blob_defects`):

| Defect | How it is made | Real-world cause |
|---|---|---|
| Connected missing regions | white noise, Gaussian blur (σ 1.5-5 px at 320×240), thresholded at the missing fraction | Structured-light / stereo / ToF cameras lose whole patches where the IR pattern is refracted through glass, absorbed by black surfaces or mirrored away by metal, not isolated pixels |
| Missing fraction 0.1-0.9 per episode | drawn per region and episode | How much a real sensor loses on the same glass changes a lot with viewing angle, background, ambient IR and the sensor model (e.g. 8-89% readings on glass in the T-ROBI / TRansPose / MJU-Glass data, `real_data/README.md`) |
| Wrong depth (0-30% of the remaining pixels, smooth error of 5-30 cm) | a second blob mask; depth = surface behind (glass) or the surface itself (absorbers, table) + a smooth random error | Multipath and mixed returns: ToF / structured light reports depths that are neither the glass nor the surface behind it (e.g. the L515's median error of 13 cm on glass in TRansPose) |

So a patch of missing or odd depth is not glass by its texture. The outline of a region can still show at high missing
fractions (a glass-shaped region vs. a round patch on the table); `sim/probe.py` measures what is left.

**Lever D** (shape matters) needs no code: the five objects have different grasp heights (`OBJECTS`), so a policy that
mistakes the object type closes at the wrong height (`wrong_height` failures). Lever A is what makes the type hard to
see in RGB.

The shared hard scene (`hard_base.yaml`) has the three properties of the main scenario: several objects (1-3 opaque
distractors: mug, box, can, bowl; held out: opaque bottle, opaque cup), a non-uniform background (solid colours,
checker, wood, noise, sky-blue tables; held out: stripes, marble, white) with varied light (normal, dim, bright, back
light; held out: spot, dusk), and a transparent target that shows what is behind it. It also puts liquid in the tumbler,
bottle and jar (fill 0 / 0.33 / 0.5 / 0.67 of the body; cold 4-10 °C, room, warm 40-60 °C; in thermal the glass wall below
the liquid line is 0.8 · liquid + 0.2 · glass temperature; the wineglass and flask are left empty because their bowl /
bulb cannot be filled with a primitive), and uses the calibrated thermal camera with thermal crossover in 10% of the
episodes. Distractors keep 9 cm (edge to edge) from the target and 2 cm from each other; 70% are within ±1 °C of the
ambient, 30% are 5-35 °C hotter or colder.

**Thermal configs** (`sim/configs/thermal/`): `v0` is main's (relative noise, per-frame min-max). `calibrated_v1` is
fitted to TRansPose `seq_all_01` raw 14-bit FLIR frames: counts = 42.67 · radiance + 811.2 (median ~3000), noise 2.73
counts at 160×120, Gaussian blur σ 1 px, and `centered` normalization over (-174, 404) counts around the frame median
(`calibrated_v1_narrow`: (-177, 213); `calibrated_v1_fixed`: fixed (2692, 3435)), so a crossover frame stays flat
instead of being stretched to full range. `realistic_p` (default 0) puts the glass 1.25-0.76 °C below ambient as measured.

**Test sets** (`sim/configs/manifest.json`, loaded by `sim/testsets.py`) share the seeds 100000-100049: (a) in
distribution, (b) held-out backgrounds, (c) held-out lighting, (d) held-out distractors, (e) liquid in every episode
(tumbler, bottle, jar), (f) thermal crossover in every episode (|gap| < 1 °C), (g) everything held out.

`sim/preview_levels.py` renders every level as the policy sees it, `sim/teacher_check.py` runs the teacher on each
level (100% everywhere), and `sim/probe.py` checks with a small CNN whether each input combination shows where the
target is and what it is. The optional narrow camera (`camera: front_cam_narrow`, 22°, fixed direction) is off by default.

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

To train on a level, collect its demos with `--config sim/configs/L<k>.yaml` and add `task.env_runner.level=L<k>` so
the evaluation during training runs on that level's test set (a).

4. Evaluate a checkpoint, e.g. the RGB policy from the release (`rgb_5obj_epoch100_ema.ckpt`). Test scene k uses seed
   `test_start_seed + k`; other seeds give new scenes drawn the same way. For a level, add
   `--level L3 --test_set a_in_distribution --test_set g_heldout_all ...`; each test set gets `eval_log.json` and
   `episodes.json` (scene conditions and failure type of every episode).
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
