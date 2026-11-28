#!/bin/bash
# Evaluates the last checkpoint of a finished L0_vis run on L0_vis_ell: the same scenes and seeds with the contact
# model of the pick-and-place level (results/hard_scenario/contact_check/README.md). Nothing is trained.
#
#   GPU=0 bash policy/eval_contacts.sh <run> [<run> ...]
#
# <run>  run name; its checkpoint is $RUNS_DIR/<run>/checkpoints/latest.ckpt (RUNS_DIR default data/runs). The
#        results go to results/hard_scenario/contact_check/<run>/<test set>/. A run that has them already is skipped.
# NEVAL (660 episodes per test set), NENVS (10), LEVEL (L0_vis_ell).
set -e
RUNS_DIR=${RUNS_DIR:-data/runs}; GPU=${GPU:-0}; NEVAL=${NEVAL:-660}; NENVS=${NENVS:-10}; LEVEL=${LEVEL:-L0_vis_ell}
DP_ROOT=${DP_ROOT:-../diffusion_policy}
SETS="j_cells11 k_cells11_heldout"
export PYTHONPATH=$PWD:$PWD/sim:$DP_ROOT:$PYTHONPATH MUJOCO_GL=${MUJOCO_GL:-egl} WANDB_MODE=${WANDB_MODE:-offline}
for NAME in "$@"; do
    OUT=results/hard_scenario/contact_check/$NAME
    if [ -f $OUT/${SETS##* }/eval_log.json ]; then echo "[$(date +%H:%M)] $NAME: already evaluated"; continue; fi
    mkdir -p $OUT
    echo "[$(date +%H:%M)] $NAME: evaluating on $LEVEL, $SETS, $NEVAL episodes each (log $RUNS_DIR/$NAME.eval_ell.log)"
    CUDA_VISIBLE_DEVICES=$GPU python policy/eval.py -c $RUNS_DIR/$NAME/checkpoints/latest.ckpt -o $OUT --level $LEVEL \
        --n_test $NEVAL --n_envs $NENVS --n_vis 0 $(for t in $SETS; do echo --test_set $t; done) \
        > $RUNS_DIR/$NAME.eval_ell.log 2>&1
    echo "[$(date +%H:%M)] $NAME: done"
done
