#!/bin/bash
# Trains the second policy of the benchmark (the action head of ACT on the benchmark's encoders, policy/act_policy.py)
# on one input-modality combination, without or with the hindsight cue.
#
#   CUDA_VISIBLE_DEVICES=0 policy/train_act.sh <arm> <mods> <zarr> [hydra overrides]
#   TRAIN="policy/train_act.sh cue" bash scripts/run_experiment.sh L0_vis rgbdt 700
#
# arm   plain  images and joints (the counterpart of the plain baselines)
#       cue    plus the hindsight cue in the conditioning, features not gated (the counterpart of the arm `cue` of
#              policy/train_cuegate.sh; task.hcue_dir, default data/hcue/L0_vis_700)
# mods  rgb | rgbd | rgbt | rgbdt (joint angles always included)
# Image encoders start from ImageNet, as those of the baselines of L0_vis do.
set -e
ARM=$1
MODS=$2
ZARR=$3
shift 3
DP_ROOT=${DP_ROOT:-../diffusion_policy}

case $MODS in
    rgb)   DROP="~task.shape_meta.obs.depth ~task.shape_meta.obs.thermal" ;;
    rgbd)  DROP="~task.shape_meta.obs.thermal" ;;
    rgbt)  DROP="~task.shape_meta.obs.depth" ;;
    rgbdt) DROP="" ;;
    *) echo "mods must be one of rgb | rgbd | rgbt | rgbdt"; exit 1 ;;
esac
case $ARM in
    plain) CFG=train_glass_act_image ;;
    cue)   CFG=train_glass_act_cue_image ;;
    *) echo "arm must be plain or cue"; exit 1 ;;
esac

export PYTHONPATH=$PWD:$PWD/sim:$DP_ROOT:$PYTHONPATH
export MUJOCO_GL=${MUJOCO_GL:-egl}
export SDL_VIDEODRIVER=dummy
export WANDB_MODE=${WANDB_MODE:-offline}

python $DP_ROOT/train.py --config-dir=policy/config --config-name=$CFG \
    task.zarr_path=$ZARR exp_name=act_${ARM}_${MODS} policy.obs_encoder.rgb_model.weights=IMAGENET1K_V1 $DROP "$@"
