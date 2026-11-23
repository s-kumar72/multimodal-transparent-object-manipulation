#!/bin/bash
# Trains the control of hindsight cue gating with an auxiliary loss (policy/aux_grasp_policy.py): the plain Diffusion
# Policy, its encoder also trained to predict the hindsight grasp point of the episode. No cue in the conditioning.
#
#   CUDA_VISIBLE_DEVICES=0 policy/train_auxgrasp.sh <mods> <zarr> [hydra overrides]
#   TRAIN=policy/train_auxgrasp.sh bash scripts/run_experiment.sh L0_vis rgbdt 700
#
# mods  rgb | rgbd | rgbt | rgbdt (joint angles always included)
# The grasp points of the demo set must exist (python policy/hindsight_cue.py <zarr> --out <dir> --mode label;
# task.hcue_dir=<dir>, default data/hcue/L0_vis_700_label). Image encoders start from ImageNet as the baselines of
# L0_vis do.
set -e
MODS=$1
ZARR=$2
shift 2
DP_ROOT=${DP_ROOT:-../diffusion_policy}

case $MODS in
    rgb)   DROP="~task.shape_meta.obs.depth ~task.shape_meta.obs.thermal" ;;
    rgbd)  DROP="~task.shape_meta.obs.thermal" ;;
    rgbt)  DROP="~task.shape_meta.obs.depth" ;;
    rgbdt) DROP="" ;;
    *) echo "mods must be one of rgb | rgbd | rgbt | rgbdt"; exit 1 ;;
esac

export PYTHONPATH=$PWD:$PWD/sim:$DP_ROOT:$PYTHONPATH
export MUJOCO_GL=${MUJOCO_GL:-egl}
export SDL_VIDEODRIVER=dummy
export WANDB_MODE=${WANDB_MODE:-offline}

python $DP_ROOT/train.py --config-dir=policy/config --config-name=train_glass_auxgrasp_unet_image \
    task.zarr_path=$ZARR exp_name=auxgrasp_${MODS} policy.obs_encoder.rgb_model.weights=IMAGENET1K_V1 $DROP "$@"
