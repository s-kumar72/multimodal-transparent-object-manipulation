#!/bin/bash
# Trains one arm of hindsight cue gating (policy/cue_gate_encoder.py) on one input-modality combination.
#
#   CUDA_VISIBLE_DEVICES=0 policy/train_cuegate.sh <arm> <mods> <zarr> [hydra overrides]
#   TRAIN="policy/train_cuegate.sh full" bash scripts/run_experiment.sh L0_vis rgbdt 700
#
# arm   full   the cue in the conditioning and the sensors' features gated by it (hard gate)
#       cue    the cue in the conditioning, features not gated
#       gate   features gated, the policy is told which sensors are on but not where the grasp point is
#       top1   as full, only the most trusted sensor's features on
#       soft   as full, features multiplied by the gate instead of switched
#       blind  no images at all: cue and joints only (mods is ignored)
# mods  rgb | rgbd | rgbt | rgbdt (joint angles always included)
# The cue of the demo set must exist (python policy/hindsight_cue.py <zarr> --out <dir>; task.hcue_dir=<dir>, default
# data/hcue/L0_vis_700). Image encoders start from ImageNet as the baselines of L0_vis do.
# CUES=2 (pick-and-place): the task glass_pick_hcue2, the grasp cue and the place cue (task.hcue_dir, task.hcue_dir2);
# every arm then has both cues, and a sensor's gate is the larger of its two r.
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

ENC=policy.obs_encoder
case $ARM in
    full)  ARM_OVR="$ENC.gate=hard $ENC.cue=true" ;;
    cue)   ARM_OVR="$ENC.gate=none $ENC.cue=true" ;;
    gate)  ARM_OVR="$ENC.gate=hard $ENC.cue=false" ;;
    top1)  ARM_OVR="$ENC.gate=top1 $ENC.cue=true" ;;
    soft)  ARM_OVR="$ENC.gate=soft $ENC.cue=true" ;;
    blind) ARM_OVR="$ENC.gate=none $ENC.cue=true"       # no image keys at all: nothing to load, encode or gate
           DROP="~task.shape_meta.obs.rgb ~task.shape_meta.obs.depth ~task.shape_meta.obs.thermal" ;;
    *) echo "arm must be one of full | cue | gate | top1 | soft | blind"; exit 1 ;;
esac

export PYTHONPATH=$PWD:$PWD/sim:$DP_ROOT:$PYTHONPATH
export MUJOCO_GL=${MUJOCO_GL:-egl}
export SDL_VIDEODRIVER=dummy
export WANDB_MODE=${WANDB_MODE:-offline}

[ "${CUES:-1}" = 2 ] && TASK="task=glass_pick_hcue2" || TASK=""
python $DP_ROOT/train.py --config-dir=policy/config --config-name=train_glass_cuegate_unet_image $TASK \
    task.zarr_path=$ZARR exp_name=${ARM}_${MODS} policy.obs_encoder.rgb_model.weights=IMAGENET1K_V1 $DROP $ARM_OVR "$@"
