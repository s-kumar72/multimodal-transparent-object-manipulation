# Running the five-object experiments on PSC Bridges-2

Everything below runs on Bridges-2 from your own account. Paths assume `PROJECT_DIR=/ocean/projects/cis250019p/$USER`;
if your allocation is a different project, `export PROJECT_DIR=/ocean/projects/<project>/$USER` first and pass it to
the jobs (`--export=ALL,...` already forwards your environment).

Results of the RGB run and a description of the environment are in `results/rgb_5objects/README.md`.

## 1. Clone the repo and install (once, about 30 min)

```bash
export PROJECT_DIR=/ocean/projects/cis250019p/$USER
mkdir -p $PROJECT_DIR/transparent && cd $PROJECT_DIR/transparent
git clone -b sim/multi-object https://github.com/s-kumar72/multimodal-transparent-object-manipulation.git
cd multimodal-transparent-object-manipulation
bash scripts/bridges2/setup.sh
```
`setup.sh` clones `real-stanford/diffusion_policy` next to the repo and builds a Python 3.11 conda env in
`$PROJECT_DIR/transparent/envs/dp` with torch 2.11 (CUDA 12.6), the MuJoCo sim, and Diffusion Policy's dependencies.
Most of the time goes into downloading torch. It ends by printing the versions and `setup done`.

## 2. Get the data and the trained checkpoint (about 5 min)

```bash
R=https://github.com/s-kumar72/multimodal-transparent-object-manipulation/releases/download/multi-object-v0.1
mkdir -p data/checkpoints
curl -L -o data/sim_demos_5obj.zarr.zip $R/sim_demos_5obj.zarr.zip                         # 489 MB, 500 demos
curl -L -o data/checkpoints/rgb_5obj_epoch100_ema.ckpt $R/rgb_5obj_epoch100_ema.ckpt      # 1.15 GB, RGB policy
```
To make the demos yourself instead (e.g. after changing the sim), run on a laptop from the repo root, then copy the
zip to `data/` on Bridges-2:
```bash
python sim/collect_demos.py --n 500 --workers 6 --out data/sim_demos_5obj.zarr
python sim/inspect_dataset.py data/sim_demos_5obj.zarr
python -c "import zarr, zipfile; d = zarr.ZipStore('data/sim_demos_5obj.zarr.zip', mode='w', compression=zipfile.ZIP_STORED); zarr.copy_store(zarr.DirectoryStore('data/sim_demos_5obj.zarr'), d); d.close()"
```
Use the zip, not the zarr directory: `/ocean` writes the zarr's 93k small files very slowly, and Diffusion Policy opens
a path ending in `.zip` directly.

## 3. Evaluate the trained checkpoint (about 10 min, 0.15 SU)

All `sbatch` commands are run from the repo root.
```bash
sbatch --export=ALL,CKPT=data/checkpoints/rgb_5obj_epoch100_ema.ckpt scripts/bridges2/eval.sbatch
```
The log `logs/dp-eval-<jobid>.out` ends with a line like
`test/mean_score 0.940  per object {'bottle': '10/10', 'flask': '10/10', 'jar': '10/10', 'tumbler': '7/10', 'wineglass': '10/10'}`
(a test run on 2026-10-24; the same checkpoint scored 0.96 during training). The policy samples with random noise, so a
rerun can differ by an episode or two. Results and videos go to `data/eval/<checkpoint>_seed100000/`.
`SEED=200000` evaluates on 50 new scenes, and `NTEST` changes the number of scenes.

## 4. Quick check before a long run (about 10 min)

Two very short epochs with all three image inputs, an evaluation on 5 scenes, and no checkpoints:
```bash
sbatch -t 0:40:00 --export=ALL,VARIANT=rgbdt,NTEST=5,EXTRA="training.debug=True checkpoint.topk.k=0 checkpoint.save_last_ckpt=False" scripts/bridges2/train.sbatch
```
It should end without a traceback, after two `test/mean_score` lines (near 0, since the policy has barely trained).

## 5. Train

```bash
for v in rgb rgbd rgbt rgbdt; do sbatch --export=ALL,VARIANT=$v scripts/bridges2/train.sbatch; done
squeue -u $USER
```
Each variant takes about 3.5 h on a 16 GB V100 (about 3.5 SU) with 150 epochs and an evaluation on 50 scenes every
25 epochs. `scripts/bridges2/train.sbatch` lists the knobs (`EPOCHS`, `NTEST`, `NENVS`, `EXTRA`). To use an H100 instead
(2 SU/h, often a long queue): `sbatch --gpus=h100-80:1 --export=ALL,VARIANT=rgb,NENVS= scripts/bridges2/train.sbatch`.

## 6. Read the results

- `logs/dp-glass-<jobid>.out`: every 25 epochs a line `test/mean_score 0.xxx  per object {...}`.
- `data/outputs/<date>/<time>_..._<variant>/`: `logs.json.txt` (losses and scores), rollout videos in `media/`,
  checkpoints (4.6 GB each, the 5 best plus `latest.ckpt`), and the offline wandb run (`wandb sync` it to upload).

## Troubleshooting

- **The job sits in the queue with reason `Priority`.** H100s often have a long queue; the V100 default usually starts
  within minutes. `squeue -p GPU-shared -t PD -o %b | sort | uniq -c` shows how many jobs wait for each GPU type.
- **`setup.sh` fails while building gym 0.21.** It needs `pip<24.1`, `setuptools<66` and `wheel<0.40`, which the
  script installs first; rerun it from a clean env (`rm -rf $PROJECT_DIR/transparent/envs/dp`).
- **Copying files from a Mac.** macOS `tar` adds `._*` files; set `COPYFILE_DISABLE=1` before `tar`.
