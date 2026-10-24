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

### Scene randomization

`GlassPickEnv(config=..., split="train" | "test")` randomizes the table background, lighting, liquid in the glass, and opaque distractor objects.
Configs live in `sim/configs/scene/` and only list what they change from `DEFAULT_SCENE` in `sim/config.py`; the default config is the original fixed scene (same physics and images for the same seed).
Each factor has a `train` pool and a disjoint `test` pool (held-out values), and the same `(seed, config, split)` always gives the same scene.

| Factor | Train pool (`randomized_v1`) | Test pool | Thermal |
|---|---|---|---|
| Table background | plain brown / gray, B&W checker, stripes, light wood, fine noise, glass-like blue | color checker, red stripes, dark wood, coarse noise, glass-like white | own temperature; emissivity varies a little by material |
| Lighting | normal, dim, bright (+ random direction / intensity) | backlight, spot | none (thermal ignores lighting) |
| Liquid | fill 0 / 0.33 / 0.5 / 0.67 of the glass, cold (5–12 C), room (t_env ± 1), warm (40–60 C) | same | glass wall below the surface is 0.8 · liquid + 0.2 · glass temperature, so a level line shows |
| Distractors (0–3) | mug, box, can | bowl, bottle, opaque cup | mostly near t_env, 30% hot (35–60 C) or cold (5–12 C) |

MuJoCo has no fluids or refraction: the liquid is a fixed cylinder inside the glass (a "glass of water" to pick, not pouring), and it is transparent in depth like the glass.
The textures in `sim/assets/gen3_lite/textures/` are made by `python sim/make_textures.py` (fixed seed).

Check that the teacher still succeeds (physics only, no rendering or GPU; aim for >= 80%):
```bash
python sim/teacher_check.py --config randomized_v1 --split train --n 100 --workers 4
python sim/teacher_check.py --config randomized_v1 --split test --n 100 --seed 100000 --workers 4
```
On a shared login node CPU time is throttled, so run longer checks on a compute node.

### Dataset format

Zarr stores in Diffusion Policy's layout; loads with `ReplayBuffer.copy_from_path(path, keys=[...]). There is one row per timestep at **10 Hz** with all episodes back to back.

| Key | Shape · dtype | Meaning |
|---|---|---|
| `data/rgb` | (N, 96, 96, 3) uint8 | Color image from `front_cam` |
| `data/depth` | (N, 96, 96, 1) uint8 | 0–2 m mapped to 0–255; 0 = no reading (holes on glass) |
| `data/thermal` | (N, 96, 96, 1) uint8 | Simulated thermal, normalized per frame (relative, not °C) |
| `data/thermal_raw` | (N, 96, 96, 1) uint16 | Raw-like 14-bit counts (0–16383) before normalization: `80 · band radiance + 4000` (v0 placeholder, ~65 counts/°C near 22 °C), with blur and noise applied in counts |
| `data/joints` | (N, 7) float32 | 6 arm joint angles (rad) + gripper (0 open – 0.85 closed) |
| `data/action` | (N, 7) float32 | Teacher's 6 absolute joint targets (rad) + gripper target |
| `meta/episode_ends` | (episodes,) int64 | Index one past the last step of each episode |

`data/thermal` is now min-max normalized from the integer raw counts, so it can differ from datasets made before `data/thermal_raw` existed by at most 1 gray level (about 10% of pixels).

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
