#!/bin/bash
# Measures training speed (it/s) and evaluation time per modality on this machine, to estimate a full run.
# Each variant: evaluation of N_EVAL test episodes (at epoch 0, untrained policy, so every episode runs all 120 steps:
# the worst case) and STEPS training steps. About 5 min per variant on a T4.
#   bash scripts/aws/benchmark.sh data/sim_demos_L0.zarr [rgb rgbd rgbdt]
# LEVEL=L4b evaluates on that level's test set (a) (rendering cost depends on the scene; training speed does not).
# Prints one line per variant; logs go to /opt/dlami/nvme/work/bench/.
set -e
ZARR=$1
shift
VARIANTS=${@:-rgb rgbd rgbdt}
STEPS=${STEPS:-300}
N_EVAL=${N_EVAL:-10}
OUT=/opt/dlami/nvme/work/bench
mkdir -p $OUT
for v in $VARIANTS; do
    log=$OUT/$v.log
    start=$(date +%s)
    policy/train.sh $v $ZARR training.num_epochs=1 training.max_train_steps=$STEPS training.max_val_steps=5 \
        task.env_runner.n_train=0 task.env_runner.n_test=$N_EVAL task.env_runner.n_test_vis=0 \
        checkpoint.topk.k=0 checkpoint.save_last_ckpt=False ${LEVEL:+task.env_runner.level=$LEVEL} \
        hydra.run.dir=$OUT/run_$v > $log 2>&1
    total=$(( $(date +%s) - start ))
    # tqdm's last report of the loop, e.g. "Training epoch 0:  92%|...| 300/326 [01:02<00:05,  4.8it/s]"
    # (the loop stops at max_train_steps, so it never reaches 100%)
    train=$(grep -ao "Training epoch 0:[^]]*]" $log | tail -1 | grep -oE "[0-9.]+it/s|[0-9.]+s/it")
    evalt=$(grep -ao "Eval GlassPickRunner 1:[^]]*]" $log | tail -1 | grep -oE "\[[0-9:]+" | tr -d '[')
    echo "$v: train $train, eval of $N_EVAL episodes $evalt (mm:ss), whole run ${total}s"
done
