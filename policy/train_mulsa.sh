#!/bin/bash
# Diffusion Policy with self-attention fusion of the sensors (policy/attn_fusion_encoder.py, the fusion of MulSA:
# Li et al., CoRL 2022), a baseline from the literature. Arguments and variants as policy/train.sh; nothing else
# differs from it.
#   CUDA_VISIBLE_DEVICES=0 policy/train_mulsa.sh rgbdt data/sim_demos.zarr [extra hydra overrides]
set -e
VARIANT=$1
ZARR=$2
shift 2
DP_ROOT=${DP_ROOT:-../diffusion_policy}

case $VARIANT in
    rgbd)  DROP="~task.shape_meta.obs.thermal" ;;
    rgbt)  DROP="~task.shape_meta.obs.depth" ;;
    rgbdt) DROP="" ;;
    *) echo "variant must be one of rgbd | rgbt | rgbdt"; exit 1 ;;
esac

export PYTHONPATH=$PWD:$PWD/sim:$DP_ROOT:$PYTHONPATH
export MUJOCO_GL=${MUJOCO_GL:-egl}
export SDL_VIDEODRIVER=dummy
export WANDB_MODE=${WANDB_MODE:-offline}

python $DP_ROOT/train.py --config-dir=policy/config --config-name=train_glass_mulsa_unet_image \
    task.zarr_path=$ZARR exp_name=mulsa_$VARIANT $DROP "$@"
