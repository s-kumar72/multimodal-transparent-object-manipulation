#!/bin/bash
# One experiment end to end on any machine with a GPU: demos (if missing) -> train -> evaluate the last checkpoint on
# test sets (a) and (g) -> results in results/hard_scenario/<GROUP>/<run>/. Nothing is committed.
#   bash scripts/run_experiment.sh <level> <variant> <n_demos>         e.g.  bash scripts/run_experiment.sh L4b rgbdt 500
# Run from the repo root, with the Diffusion Policy repo at ../diffusion_policy (or set DP_ROOT).
# Rerunning the same command resumes an unfinished training run (Diffusion Policy reloads checkpoints/latest.ckpt from
# the same run directory) and skips the steps that are already done.
#
# Knobs (environment): EPOCHS (default 101 = 100 epochs + the evaluation and checkpoint at epoch 100), NTEST (50),
# NENVS (envs rendered at once, 10), WORKERS (demo collection processes, 4), GROUP (results folder, diagnosis),
# RUNS_DIR (training outputs and checkpoints, data/runs; ~5 GB per run), EXTRA (more hydra overrides),
# TAG (suffix of the run name for runs that differ only by EXTRA, e.g. TAG=imagenet), GPU (CUDA_VISIBLE_DEVICES, 0).
# Several runs can start at once on different GPUs: a lock makes the first one collect a missing demo set and the
# others wait for it.
set -e
LEVEL=$1; VARIANT=$2; NDEMOS=$3
[ -n "$NDEMOS" ] || { echo "usage: bash scripts/run_experiment.sh <level> <variant> <n_demos>"; exit 1; }
EPOCHS=${EPOCHS:-101}; NTEST=${NTEST:-50}; NENVS=${NENVS:-10}; WORKERS=${WORKERS:-4}
GROUP=${GROUP:-diagnosis}; RUNS_DIR=${RUNS_DIR:-data/runs}; GPU=${GPU:-0}
DP_ROOT=${DP_ROOT:-../diffusion_policy}
export PYTHONPATH=$PWD:$PWD/sim:$DP_ROOT:$PYTHONPATH MUJOCO_GL=${MUJOCO_GL:-egl} WANDB_MODE=${WANDB_MODE:-offline}

NAME=${LEVEL}_${VARIANT}_${NDEMOS}demos${TAG:+_$TAG}
ZARR=results/data/sim_demos_${LEVEL}_${NDEMOS}.zarr
RUN=$RUNS_DIR/$NAME
OUT=results/hard_scenario/$GROUP/$NAME
mkdir -p $RUNS_DIR $OUT

# 1. Demos: seeds 0..NDEMOS-1, so the first 200 of a 500-demo set are the 200-demo set
mkdir -p results/data
(
    flock 9
    if [ ! -f $ZARR/meta/episode_ends/.zarray ]; then
        echo "[$(date +%H:%M)] $NAME: collecting $NDEMOS demos -> $ZARR"
        python sim/collect_demos.py --n $NDEMOS --workers $WORKERS --config sim/configs/$LEVEL.yaml --out $ZARR \
            > $RUN.demos.log 2>&1
        grep "teacher success" $RUN.demos.log
    fi
) 9> $ZARR.lock

# 2. Training (resumes from $RUN/checkpoints/latest.ckpt if a previous attempt was interrupted)
# Done = the last epoch's evaluation is logged (its step lines also carry the epoch number, so check for the score)
if ! grep "\"epoch\": $((EPOCHS - 1))," $RUN/logs.json.txt 2>/dev/null | grep -q "test/mean_score"; then
    echo "[$(date +%H:%M)] $NAME: training $EPOCHS epochs -> $RUN (log $RUN.log)"
    CUDA_VISIBLE_DEVICES=$GPU DP_ROOT=$DP_ROOT policy/train.sh $VARIANT $ZARR training.num_epochs=$EPOCHS \
        task.env_runner.n_test=$NTEST task.env_runner.n_envs=$NENVS task.env_runner.level=$LEVEL \
        checkpoint.topk.k=1 hydra.run.dir=$RUN ${EXTRA:-} >> $RUN.log 2>&1
fi
cp $RUN/logs.json.txt $OUT/train_logs.json.txt
python policy/summarize_runs.py curve $OUT/train_logs.json.txt $OUT/curve.json

# 3. Evaluation of the last checkpoint on test sets (a) and (g)
if [ ! -f $OUT/g_heldout_all/eval_log.json ]; then
    echo "[$(date +%H:%M)] $NAME: evaluating test sets (a) and (g) (log $RUN.eval.log)"
    CUDA_VISIBLE_DEVICES=$GPU python policy/eval.py -c $RUN/checkpoints/latest.ckpt -o $OUT --level $LEVEL \
        --n_test $NTEST --n_envs $NENVS --n_vis 0 --test_set a_in_distribution --test_set g_heldout_all \
        > $RUN.eval.log 2>&1
fi
echo "[$(date +%H:%M)] $NAME: done -> $OUT"
python policy/summarize_runs.py table $OUT
