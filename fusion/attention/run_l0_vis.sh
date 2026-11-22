#!/bin/bash
# The learned attention weighting on the visibility benchmark L0_vis (results/hard_scenario/grid_bench/README.md):
# segmentation pre-training on L0_vis frames, then the policy runs of fusion/attention/ablations.MD on that level.
# Run from the repo root, Diffusion Policy at ../diffusion_policy (or DP_ROOT):
#
#   DRY=1 bash fusion/attention/run_l0_vis.sh                        # print every command, run nothing
#   STAGES="frames seg" GPUS=0 bash fusion/attention/run_l0_vis.sh   # the pre-training only
#   STAGES=policy GPUS="0 2 3" bash fusion/attention/run_l0_vis.sh   # the policy runs, one per GPU at a time
#
# Stages (STAGES, default "frames seg policy"; finished work is skipped):
#   frames  renders the 96x96 rgb / depth / thermal frames of the 700 training scenes of L0_vis with the simulator's
#           exact glass mask (fusion/attention/collect_sim.py), plus 40 validation scenes and 110 episodes of each
#           test set -> DATA/sim
#   seg     trains the segmentation run(s) SEG_RUNS with fusion/attention/train.sh (mix_ws: sim frames + real data,
#           learned per-sensor weights) on the first GPU of GPUS and writes results/fusion/summary_<DATA folder>.md
#   policy  Diffusion Policy with the fusion encoder started from the run SEG_INIT: the runs RUNS of
#           fusion/attention/run_ablations.sh with the protocol of the L0_vis baselines (700 demos, 101 epochs, last
#           checkpoint, rollouts during training on j_cells11 with 55 episodes, final evaluation on j_cells11 and
#           k_cells11_heldout with 660 episodes each and the sensor weights saved per step)
#           -> results/hard_scenario/GROUP/<run>/
#
# Settings: DATA (data/fusion_vis), SEG_RUNS (mix_ws), SEG_INIT (mix_ws), RUNS ("fusion fusion_kp fusion_static" = P0,
# P9 and P5 of ablations.MD), SEEDS ("42"), GPUS ("0 1 2 3"), PER_GPU (1), GROUP (fusion_vis), WORKERS (rendering
# processes, 8), EGL_OF (see run_ablations.sh), DRY (1: print only).
#
# What these runs use that the plain baselines do not: the simulator's glass masks and its flag for the frames in which
# thermal does not show the glass (the gate loss of mix_ws), both in the segmentation pre-training only. The policy
# training and the evaluation see sensor images and joint values, as every baseline.
# Success per cell of a finished run: python policy/summarize_cells.py results/hard_scenario/GROUP/<run>
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
STAGES=${STAGES:-"frames seg policy"}
DATA=${DATA:-data/fusion_vis}
SEG_RUNS=${SEG_RUNS:-mix_ws}
SEG_INIT=${SEG_INIT:-mix_ws}
RUNS=${RUNS:-"fusion fusion_kp fusion_static"}
SEEDS=${SEEDS:-42}
GPUS=${GPUS:-"0 1 2 3"}
PER_GPU=${PER_GPU:-1}
GROUP=${GROUP:-fusion_vis}
WORKERS=${WORKERS:-8}
DRY=${DRY:-0}
LEVEL=L0_vis
TEST_SETS="j_cells11 k_cells11_heldout"
DP_ROOT=${DP_ROOT:-../diffusion_policy}
export PYTHONPATH=$PWD:$PWD/sim:$DP_ROOT:${PYTHONPATH:-} MUJOCO_GL=${MUJOCO_GL:-egl} WANDB_MODE=${WANDB_MODE:-offline}
has() { [[ " $STAGES " == *" $1 "* ]]; }
run() { if [ "$DRY" = 1 ]; then echo "$*"; else echo "+ $*"; eval "$*"; fi; }

if has frames; then
    echo "== frames: $LEVEL -> $DATA/sim"
    run "python -m fusion.attention.collect_sim --levels $LEVEL --n_train 700 --n_val 40 --n_test 110" \
        "--test_sets $TEST_SETS --stride 6 --workers $WORKERS --out $DATA/sim"
fi

if has seg; then
    echo "== seg: $SEG_RUNS on GPU ${GPUS%% *}"
    run "DATA=$DATA SIM_LEVELS=$LEVEL STAGES='real seg summary' RUNS='$SEG_RUNS' GPUS=${GPUS%% *} WORKERS=$WORKERS" \
        "bash fusion/attention/train.sh"
fi

if has policy; then
    echo "== policy: $RUNS, seeds $SEEDS, from $DATA/runs/$SEG_INIT/best.pt"
    echo "   scripts/run_experiment.sh settings: TEST_SETS=\"$TEST_SETS\" NTEST=55 NEVAL=660 TOPK=0"
    [ "$DRY" = 1 ] || [ -f "$DATA/runs/$SEG_INIT/best.pt" ] || { echo "no $DATA/runs/$SEG_INIT/best.pt (stage seg)"; exit 1; }
    LEVEL=$LEVEL NDEMOS=700 GROUP=$GROUP SEEDS="$SEEDS" P0_SEEDS="$SEEDS" ONLY="$RUNS" ONLY_VARIANT=rgbdt \
        SEG_CKPT=$DATA/runs/$SEG_INIT/best.pt BASE_EXTRA="task.env_runner.test_set=j_cells11" \
        TEST_SETS="$TEST_SETS" NTEST=55 NEVAL=660 TOPK=0 GPUS="$GPUS" PER_GPU=$PER_GPU DRY=$DRY \
        bash fusion/attention/run_ablations.sh
fi
