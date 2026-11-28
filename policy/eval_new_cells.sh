#!/bin/bash
# Evaluates the last checkpoint of finished L0_vis runs on the test sets whose visibility combinations are not in the
# training set (l_cells15_new, m_cells15_new_heldout; results/hard_scenario/cue_gate/README.md, round 3a).
#
#   GPU=0 bash policy/eval_new_cells.sh <group>/<run> [<group>/<run> ...]
#
# <group>/<run>  the run's folder under results/hard_scenario/ (the new results go to <test set>/ in it); its
#                checkpoint is $RUNS_DIR/<run>/checkpoints/latest.ckpt (RUNS_DIR default data/runs)
# A run whose training and evaluation have not finished (no k_cells11_heldout/eval_log.json) is skipped, and so is one
# that has the new results already. NEVAL (600: 40 episodes per cell and test set), NENVS (10).
set -e
RUNS_DIR=${RUNS_DIR:-data/runs}; GPU=${GPU:-0}; NEVAL=${NEVAL:-600}; NENVS=${NENVS:-10}
DP_ROOT=${DP_ROOT:-../diffusion_policy}
SETS="l_cells15_new m_cells15_new_heldout"
export PYTHONPATH=$PWD:$PWD/sim:$DP_ROOT:$PYTHONPATH MUJOCO_GL=${MUJOCO_GL:-egl}
for R in "$@"; do
    OUT=results/hard_scenario/$R; NAME=${R##*/}
    if [ ! -f $OUT/k_cells11_heldout/eval_log.json ]; then echo "[$(date +%H:%M)] $NAME: not finished, skipped"; continue; fi
    if [ -f $OUT/${SETS##* }/eval_log.json ]; then echo "[$(date +%H:%M)] $NAME: already evaluated"; continue; fi
    echo "[$(date +%H:%M)] $NAME: evaluating $SETS, $NEVAL episodes each (log $RUNS_DIR/$NAME.eval_new.log)"
    CUDA_VISIBLE_DEVICES=$GPU python policy/eval.py -c $RUNS_DIR/$NAME/checkpoints/latest.ckpt -o $OUT --level L0_vis \
        --n_test $NEVAL --n_envs $NENVS --n_vis 0 $(for t in $SETS; do echo --test_set $t; done) \
        > $RUNS_DIR/$NAME.eval_new.log 2>&1
    echo "[$(date +%H:%M)] $NAME: done"
done
