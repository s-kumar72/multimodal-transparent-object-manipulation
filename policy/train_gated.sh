#!/bin/bash
# Trains one arm of the cue-gating experiment on one input-modality combination.
#
#   CUDA_VISIBLE_DEVICES=0 policy/train_gated.sh <arm> <mods> <zarr> [hydra overrides]
#
# arm   static      shared crop + depth mask + encoder init, plain concat, no cue       (strong static baseline)
#       static_cue  the same, cue vector appended to the conditioning, no gate          ("cue as input" control)
#       gate        cue-driven gate (gate_input=cue), w in cond, e_miss, targeted dropout, gate + aux losses (v1)
#       gate_feat   gate on features only (no cue)                                       (learned-only gate)
#       gate_both   gate on cue + features                                               (hybrid)
# mods  rgb | rgbd | rgbt | rgbdt (joint angles always included)
#
# Corruption during training is on for every arm (task glass_pick_cue); turn it off with task.dataset.augment=null.
# Train on a hard-scenario level with its demos:  ... data/sim_demos_L4b.zarr task.env_runner.level=L4b
# Thermal encoder from encoder/thermal_pretrain:   policy.obs_encoder.init_weights.thermal=data/thermal_resnet18.pt
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
    static)     ARM_OVR="$ENC._target_=policy.shared_crop_encoder.SharedCropObsEncoder ~task.shape_meta.obs.cue
                         ~$ENC.cue_key ~$ENC.gate_input ~$ENC.cross_modal_cue ~$ENC.gate_hidden ~$ENC.gate_act
                         ~$ENC.w_in_cond ~$ENC.cue_in_cond ~$ENC.missing ~$ENC.dropout_mode ~$ENC.dropout_p
                         ~$ENC.dropout_p_other ~$ENC.rgb_gate_bias policy.aux_weight=0 policy.gate_loss_weight=0" ;;
    static_cue) ARM_OVR="$ENC._target_=policy.shared_crop_encoder.SharedCropObsEncoder
                         ~$ENC.cue_key ~$ENC.gate_input ~$ENC.cross_modal_cue ~$ENC.gate_hidden ~$ENC.gate_act
                         ~$ENC.w_in_cond ~$ENC.cue_in_cond ~$ENC.missing ~$ENC.dropout_mode ~$ENC.dropout_p
                         ~$ENC.dropout_p_other ~$ENC.rgb_gate_bias policy.aux_weight=0 policy.gate_loss_weight=0" ;;
    gate)       ARM_OVR="" ;;
    gate_feat)  ARM_OVR="$ENC.gate_input=feature" ;;
    gate_both)  ARM_OVR="$ENC.gate_input=both" ;;
    *) echo "arm must be one of static | static_cue | gate | gate_feat | gate_both"; exit 1 ;;
esac

export PYTHONPATH=$PWD:$PWD/sim:$DP_ROOT:$PYTHONPATH
export MUJOCO_GL=${MUJOCO_GL:-egl}
export SDL_VIDEODRIVER=dummy
export WANDB_MODE=${WANDB_MODE:-offline}

python $DP_ROOT/train.py --config-dir=policy/config --config-name=train_glass_gated_unet_image \
    task.zarr_path=$ZARR exp_name=${ARM}_${MODS} $DROP $ARM_OVR "$@"
