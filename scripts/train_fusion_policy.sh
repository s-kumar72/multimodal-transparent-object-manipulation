#!/bin/bash
# Diffusion Policy with the reliability-aware fusion encoder of fusion/ (fusion/attention/policy_encoder.py, FusionObsEncoder)
# instead of one ResNet per modality; the same call as policy/train.sh, so scripts/run_experiment.sh can use it:
#   CUDA_VISIBLE_DEVICES=0 scripts/train_fusion_policy.sh rgbdt data/sim_demos.zarr \
#       policy.obs_encoder.pretrained=data/fusion/runs/mix_ws/best.pt [more hydra overrides]
#   TRAIN=scripts/train_fusion_policy.sh EXTRA="policy.obs_encoder.pretrained=..." TAG=fusion \
#       bash scripts/run_experiment.sh L0_bench rgbdt 500
# Config: fusion/attention/config/train_glass_fusion_image.yaml (everything else as policy/config/train_glass_unet_image.yaml).
# Variants: rgb | rgbd | rgbt | rgbdt (a dropped modality is treated as missing by the encoder).
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

python $DP_ROOT/train.py --config-dir=fusion/attention/config --config-name=train_glass_fusion_image \
    task.zarr_path=$ZARR exp_name=fusion_$VARIANT $DROP "$@"
