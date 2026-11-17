#!/bin/bash
# The diagnosis experiments in results/hard_scenario/diagnosis/README.md: ten rgbdt/rgbd runs on four GPUs, one queue
# per GPU (about 4-5 h on A6000s). Each queue runs its experiments one after another; a failed run is logged and the
# queue moves on. Rerunning the script resumes unfinished runs and skips finished ones (scripts/run_experiment.sh).
#   bash scripts/run_diagnosis.sh            GPUS="0 1 2 3" by default; logs in data/runs/queue_<gpu>.log
set -u
GPUS=(${GPUS:-0 1 2 3})
IMAGENET="policy.obs_encoder.rgb_model.weights=IMAGENET1K_V1"
export GROUP=diagnosis
mkdir -p data/runs

run() {  # gpu level variant demos [tag extra]
    GPU=$1 TAG=${5:-} EXTRA=${6:-} bash scripts/run_experiment.sh $2 $3 $4 || echo "!!! FAILED: $2 $3 $4 ${5:-}"
}

queue0() { run $1 L4b rgbdt 500;  run $1 L0_faint rgbdt 200; run $1 L3b rgbdt 200 imagenet "$IMAGENET"; }  # D2, 1a, 2a'
queue1() { run $1 L1 rgbdt 500;   run $1 L1_alpha02 rgbdt 200; }                                          # 1c (D1), 1b
queue2() { run $1 L3b rgbdt 500;  run $1 L1 rgbdt 200 imagenet "$IMAGENET"; }                             # 2a, 1d
queue3() { run $1 L4b rgbd 500;   run $1 L3b_mild rgbdt 200; run $1 L4b rgbdt 200 imagenet "$IMAGENET"; } # D3, 2b, P4

for i in 0 1 2 3; do
    queue$i ${GPUS[$i]} > data/runs/queue_${GPUS[$i]}.log 2>&1 &
done
echo "started 4 queues on GPUs ${GPUS[*]}; logs: data/runs/queue_<gpu>.log"
wait
echo "all queues done $(date)"
grep -h "FAILED" data/runs/queue_*.log || echo "no failed runs"
