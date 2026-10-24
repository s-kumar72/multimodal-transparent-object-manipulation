#!/bin/bash
# Trains Diffusion Policy on the sim glass-pick demos with one input-modality combination.
#
# Run from repo root with the Diffusion Policy repo cloned next to this one (or set DP_ROOT):
#   CUDA_VISIBLE_DEVICES=0 policy/train.sh rgbdt data/sim_demos.zarr [extra hydra overrides]
# Variants: rgb | rgbd | rgbt | rgbdt (joint angles are always included)
set -e
VARIANT=$1
ZARR=$2
shift 2
DP_ROOT=${DP_ROOT:-../diffusion_policy}

case $VARIANT in
    rgb)   DROP="~task.shape_meta.obs.depth ~task.shape_meta.obs.thermal" ;;
    rgbd)  DROP="~task.shape_meta.obs.thermal" ;;
    rgbt)  DROP="~task.shape_meta.obs.depth" ;;
    rgbdt) DROP="" ;;
    *) echo "variant must be one of rgb | rgbd | rgbt | rgbdt"; exit 1 ;;
esac

export PYTHONPATH=$PWD:$PWD/sim:$DP_ROOT:$PYTHONPATH
export MUJOCO_GL=${MUJOCO_GL:-egl}
export SDL_VIDEODRIVER=dummy
export WANDB_MODE=${WANDB_MODE:-offline}

python $DP_ROOT/train.py --config-dir=policy/config --config-name=train_glass_unet_image \
    task.zarr_path=$ZARR exp_name=$VARIANT $DROP "$@"
