#!/bin/bash -l
# Train the thermal encoder on GPU-shared; extra config overrides go in ARGS.
#   sbatch --export=ALL,ARGS="data.norm_mode=centered data.norm_params.range=[-70,132] out_dir=..." encoder/train.sbatch
#SBATCH -A cis250019p
#SBATCH -p GPU-shared
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --ntasks-per-node=5
#SBATCH -t 00:30:00
#SBATCH -J enc_train
#SBATCH -o /jet/home/klee30/Project_team/logs/%x_%j.out
set -e
module load anaconda3
source activate dp311
cd ${REPO:-/jet/home/klee30/Project_team/mtom-encoder}
export OMP_NUM_THREADS=4
time python -m encoder.train --config encoder/configs/transpose_seg.yaml \
    data.root=$PROJECT/transparent/transpose data.cache_dir=$PROJECT/transparent/data/transpose_cache $ARGS
