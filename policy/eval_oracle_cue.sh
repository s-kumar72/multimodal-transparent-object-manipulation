#!/bin/bash
# Evaluates the last checkpoint of a finished L0_vis run with the true grasp point in the cue (policy/eval_oracle_cue.py;
# results/hard_scenario/cue_gate/README.md, round 6c), on L0_vis and on L0_vis_ell. Nothing is trained.
#
#   GPU=0 [ORACLE_CUE_ONES=1] bash policy/eval_oracle_cue.sh <run> [<run> ...]
#
# <run>  run name; its checkpoint is $RUNS_DIR/<run>/checkpoints/latest.ckpt (RUNS_DIR default data/runs). The results
#        go to results/hard_scenario/oracle_cue/<level>/<run>/<test set>/. A run that has them already is skipped.
# ORACLE_CUE_ONES=1: confidence and gate entries of the cue set to 1 (for a policy trained on the label cue).
# NEVAL (660 episodes per test set), NENVS (10), LEVELS ("L0_vis L0_vis_ell").
set -e
RUNS_DIR=${RUNS_DIR:-data/runs}; GPU=${GPU:-0}; NEVAL=${NEVAL:-660}; NENVS=${NENVS:-10}; LEVELS=${LEVELS:-"L0_vis L0_vis_ell"}
DP_ROOT=${DP_ROOT:-../diffusion_policy}; OUT_ROOT=${OUT_ROOT:-results/hard_scenario/oracle_cue}
SETS="j_cells11 k_cells11_heldout"
export PYTHONPATH=$PWD:$PWD/sim:$DP_ROOT:$PYTHONPATH MUJOCO_GL=${MUJOCO_GL:-egl} WANDB_MODE=${WANDB_MODE:-offline}
for NAME in "$@"; do
    for LEVEL in $LEVELS; do
        OUT=$OUT_ROOT/$LEVEL/$NAME
        if [ -f $OUT/${SETS##* }/eval_log.json ]; then echo "[$(date +%H:%M)] $NAME on $LEVEL: already evaluated"; continue; fi
        mkdir -p $OUT
        echo "[$(date +%H:%M)] $NAME: the true grasp point in the cue, $LEVEL, $SETS, $NEVAL episodes each (log $RUNS_DIR/$NAME.eval_oracle_$LEVEL.log)"
        CUDA_VISIBLE_DEVICES=$GPU python policy/eval_oracle_cue.py -c $RUNS_DIR/$NAME/checkpoints/latest.ckpt -o $OUT --level $LEVEL \
            --n_test $NEVAL --n_envs $NENVS --n_vis 0 $(for t in $SETS; do echo --test_set $t; done) \
            > $RUNS_DIR/$NAME.eval_oracle_$LEVEL.log 2>&1
    done
    echo "[$(date +%H:%M)] $NAME: done"
done
