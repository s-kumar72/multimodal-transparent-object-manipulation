#!/bin/bash
# Trains one modality variant of Diffusion Policy on Bridges-2 (GPU-shared). Submit from the repo root:
#   sbatch --export=ALL,VARIANT=rgb scripts/bridges2/train.sbatch
#
# VARIANT  rgb | rgbd | rgbt | rgbdt (joint angles are always included)
# ZARR     demos, default data/sim_demos_5obj.zarr.zip
# EPOCHS   default 151 = 150 epochs plus a final evaluation and checkpoint: Diffusion Policy evaluates and saves only
#          when epoch % 25 == 0 and counts epochs from 0, so with 150 the last evaluation would be epoch 125
# NTEST    test episodes per evaluation, default 50 (10 per object, seeds 100000-100049)
# EXTRA    more hydra overrides, space separated
#
# The default GPU is a 16 GB V100 (1 SU/h, short queue). There the 54 eval envs run in 3 batches of 18 so their EGL
# contexts fit next to the model; this changes only the speed. On an H100 (2 SU/h, often a long queue):
#   sbatch --gpus=h100-80:1 --export=ALL,VARIANT=rgb,NENVS= scripts/bridges2/train.sbatch
# Quick check that everything runs (2 short epochs, 5 test episodes, no checkpoints, ~15 min):
#   sbatch -t 0:40:00 --export=ALL,VARIANT=rgbdt,NTEST=5,EXTRA="training.debug=True checkpoint.topk.k=0 checkpoint.save_last_ckpt=False" scripts/bridges2/train.sbatch
#SBATCH -J dp-glass
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-16:1
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
set -e
PROJECT_DIR=${PROJECT_DIR:-/ocean/projects/cis250019p/$USER}
ROOT=${ROOT:-$PROJECT_DIR/transparent}
ENV=${ENV_DIR:-$ROOT/envs/dp}
VARIANT=${VARIANT:-rgb}
ZARR=${ZARR:-data/sim_demos_5obj.zarr.zip}
NENVS=${NENVS-18}

if [ -f "$ENV/bin/activate" ]; then source "$ENV/bin/activate"   # a venv
else module load anaconda3; eval "$(conda shell.bash hook)"; conda activate "$ENV"; fi
cd "$ROOT/multimodal-transparent-object-manipulation"
mkdir -p logs
export DP_ROOT="$ROOT/diffusion_policy"
export MUJOCO_GL=egl                        # headless rendering on the GPU
export WANDB_MODE=offline
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # torch >= 2.6 refuses Diffusion Policy's dill checkpoints when resuming
nvidia-smi --query-gpu=name,memory.total --format=csv

policy/train.sh "$VARIANT" "$ZARR" \
    training.num_epochs=${EPOCHS:-151} \
    task.env_runner.n_test=${NTEST:-50} \
    ${NENVS:+task.env_runner.n_envs=$NENVS} \
    ${EXTRA:-}
