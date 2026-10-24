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

Collect randomized demos (train pools) with per-episode conditions in `meta/ep_*`:
```bash
python sim/collect_demos.py --n 200 --config randomized_v1 --split train --workers 5 --out data/sim_demos_rand.zarr
```

Check that the teacher still succeeds (physics only, no rendering or GPU; aim for >= 80%):
```bash
python sim/teacher_check.py --config randomized_v1 --split train --n 100 --workers 4
python sim/teacher_check.py --config randomized_v1 --split test --n 100 --seed 100000 --workers 4
```
On a shared login node CPU time is throttled, so run longer checks on a compute node.

### Test sets

A test set is (scene config, split, seeds); `GlassPickEnv(config, split).reset(seed)` rebuilds exactly the same scene, so every modality is evaluated on the same scenes.
All sets use seeds 100000.. (training uses seeds from 0), so with the same seed they differ only in the held-out factor.
Configs are in `sim/configs/scene/test/` (each starts from `randomized_v1` with `base:`).

| Set | Config | Split | Condition |
|---|---|---|---|
| a | `test/a_indist` | train | in-distribution (train pools, unseen seeds) |
| b | `test/b_heldout_background` | test | held-out backgrounds only |
| c | `test/c_heldout_lighting` | test | held-out lighting only (backlight, spot) |
| d | `test/d_heldout_distractors` | test | held-out distractors only (bowl, bottle, opaque cup) |
| e | `test/e_liquid` | train | liquid in every glass (fill 0.33 / 0.5 / 0.67 × cold / room / warm, sampled) |
| f | `test/f_crossover` | train | forced thermal crossover, \|glass − t_env\| ≤ 1 °C (thermal config `v0_crossover`) |
| g | `test/g_heldout_all` | test | all held-out factors together |

```bash
python sim/make_test_sets.py --sets a b c d g --n 50 --out data/test_sets --demos --workers 5
```
This writes `<out>/<set>/manifest.json` (config, split, seeds, teacher success per seed) and, with `--demos`, `<out>/<set>/demos.zarr` with the teacher's episodes.

### Thermal model

The scene config names a thermal config (`thermal: v0`) in `sim/configs/thermal/`; YAML files there override `DEFAULT_THERMAL` in `sim/config.py`.
It holds every thermal parameter: t_env range, glass − t_env gap and crossover probability, table / robot / distractor temperatures, emissivity ranges (glass, table, distractors), liquid mix, raw gain/offset, thermal resolution, blur, and noise in raw counts.
`v0` is the original hand-picked model and is kept unchanged; calibrated models get their own files.

`calibrated_v1` fits only the camera model (raw gain/offset, noise in counts) to TRansPose 14-bit raw thermal (seq_all_01); the glass temperature gap and crossover design stay as in v0. Blur is fixed at σ = 1 px: a fitted blur ran to the search limit (2.5 px) because it absorbed the glass size difference, so it will be refit once the camera is decided.
The fit compares image statistics on raw counts after the same 96×96 resize (count levels, |Δ|/range, |Δ|/noise, noise/range, noise, edge sharpness; CNR is reported only). It runs on "realistic" sim frames (glass slightly colder than t_env, no liquid, like TRansPose), fitting that gap together with the camera. The gap is stored as the `realistic` option, off by default (`realistic.p`, separate RNG stream).
```bash
# compare thermal configs with real data, or fit (seeded random search) and write sim/configs/thermal/calibrated_v1.yaml
python sim/thermal_calib.py --real-root $PROJECT/transparent/transpose --thermal v0 calibrated_v1
python sim/thermal_calib.py --real-root $PROJECT/transparent/transpose --fit --blur 1.0 --save calibrated_v1
```
Reports (summary table, histograms, real vs sim image grid on one fixed count range) go to `sim/out/thermal_calib/`.

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

Datasets made with this branch also store per-episode conditions as flat arrays `meta/ep_<field>` (length = episodes; DP's `ReplayBuffer` loads them as extra meta and ignores them otherwise). Read them with `load_episode_meta(root)` from `sim/dataset_meta.py`, which decodes names and returns `None` for older datasets.

| Key | Shape · dtype | Meaning |
|---|---|---|
| `meta/ep_split`, `ep_background`, `ep_lighting`, `ep_thermal_config`, `ep_thermal_norm` | (E,) int16 | ids; names in `meta.attrs["id_maps"]` |
| `meta/ep_seed` | (E,) int64 | env seed (with the config and split it rebuilds the exact scene) |
| `meta/ep_n_distractors`, `ep_distractors` | (E,) int8, (E, max_count) int16 | distractor count and ids (-1 = none) |
| `meta/ep_distractor_temps`, `ep_emis_distractors` | (E, max_count) float32 | per distractor, NaN = none |
| `meta/ep_fill_level`, `ep_liquid_temp` | (E,) float32 | liquid fill (0 = empty) and temperature (°C, NaN if empty) |
| `meta/ep_t_env`, `ep_glass_temp`, `ep_table_temp`, `ep_thermal_gap` | (E,) float32 | °C; gap = glass − t_env |
| `meta/ep_is_crossover` | (E,) bool | \|gap\| < 2 °C |
| `meta/ep_emis_glass`, `ep_emis_table` | (E,) float32 | emissivities |

`meta.attrs` also keeps the full scene config, thermal config, and split used.

`data/thermal` is now min-max normalized from the integer raw counts, so it can differ from datasets made before `data/thermal_raw` existed by at most 1 gray level (about 10% of pixels).

### Known limitations

- **Liquid level line is tiny at 96×96.** The glass covers only about 21–27 pixels (7–8 rows × 3–4 columns; default scene, seeds 0–9). When the liquid and glass temperatures differ, thermal shows a horizontal level line, but fill levels 0.33 / 0.5 / 0.67 differ by only 1–2 rows, and the 3×3 blur at 120×160 before resizing softens the edge further. Treat it as "the lower part of the glass shifts toward the liquid temperature", not as a readable fill level. Fixing this would need a different camera pose / field of view or image size, which are out of scope.
- **The top of a filled glass shows the glass temperature.** The MuJoCo glass is a closed cylinder, so its top face (seen at an angle from above) is rendered at the glass temperature. A real open glass would show the liquid surface (water emissivity ≈ 0.96, liquid temperature).
- **The glass is much smaller in the image than in real thermal data.** It covers ~0.26% of the frame, vs ~5.8% for the transparent objects in TRansPose (seq_all_01). This comes from the camera pose / field of view, not from a thermal parameter, so the thermal calibration does not try to match it. With so few pixels, a 2 px background ring around the glass is mostly blurred edge, and the glass-vs-ring CNR saturates near 5–7 in sim; `sim/thermal_calib.py` therefore also reports a CNR with the ring moved 2 px away from the edge and fits counts-per-degree independent ratios instead.
- **Little thermal structure in the background.** In TRansPose the background around the glass varies about 20× the sensor noise (other warm objects, temperature patterns). The sim background is a uniform-temperature table and floor; only the warm robot links and the distractors add structure.
- **No refraction or fluids.** MuJoCo renders the glass as a tinted, alpha-blended cylinder and the liquid as a fixed cylinder inside it (picking a glass of water, not pouring).

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
