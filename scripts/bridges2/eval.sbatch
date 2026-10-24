#!/bin/bash
# Evaluates a checkpoint in the sim on Bridges-2 and reports the success rate overall and per object.
# Submit from the repo root:
#   sbatch --export=ALL,CKPT=data/checkpoints/rgb_5obj_epoch100_ema.ckpt scripts/bridges2/eval.sbatch
#
# CKPT   checkpoint (a full training checkpoint or an EMA-only export)
# NTEST  test episodes, default 50 (10 per object)
# SEED   first test seed, default 100000 (the seeds used during training); pick another range for new scenes
# NENVS  parallel sim envs, default 18 (fits a 16 GB V100); empty = all at once
# OUT    output dir, default data/eval/<checkpoint name>_seed<SEED>
#SBATCH -J dp-eval
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-16:1
#SBATCH -t 1:00:00
#SBATCH -o logs/%x-%j.out
set -e
PROJECT_DIR=${PROJECT_DIR:-/ocean/projects/cis250019p/$USER}
ROOT=${ROOT:-$PROJECT_DIR/transparent}
ENV=${ENV_DIR:-$ROOT/envs/dp}
: "${CKPT:?set CKPT=path/to/checkpoint}"
SEED=${SEED:-100000}
NENVS=${NENVS-18}

if [ -f "$ENV/bin/activate" ]; then source "$ENV/bin/activate"
else module load anaconda3; eval "$(conda shell.bash hook)"; conda activate "$ENV"; fi
cd "$ROOT/multimodal-transparent-object-manipulation"
mkdir -p logs
export PYTHONPATH=$PWD:$PWD/sim:$ROOT/diffusion_policy:$PYTHONPATH
export MUJOCO_GL=egl
nvidia-smi --query-gpu=name,memory.total --format=csv

python policy/eval.py -c "$CKPT" -o "${OUT:-data/eval/$(basename "$CKPT" .ckpt)_seed$SEED}" \
    --n_test ${NTEST:-50} --test_start_seed $SEED ${NENVS:+--n_envs $NENVS}
