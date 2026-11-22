#!/bin/bash
# Plain Diffusion Policy with random sensor dropout in training, the usual remedy for a policy that leans on one
# sensor: each image of a training sample (rgb / depth / thermal, all its time steps) is blank with probability P
# (SENSOR_DROPOUT_P, default 0.3), independently, and at least one stays. Nothing else differs from policy/train.sh;
# validation and evaluation see all sensors.
#   CUDA_VISIBLE_DEVICES=0 policy/train_dropout.sh rgbdt data/sim_demos.zarr [extra hydra overrides]
set -e
P=${SENSOR_DROPOUT_P:-0.3}
exec "$(dirname "$0")/train.sh" "$@" "+task.dataset.augment={thermal_contrast_p:0.0,depth_hole_p:0.0,sensor_dropout_p:$P}"
