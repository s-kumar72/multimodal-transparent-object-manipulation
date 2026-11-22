#!/bin/bash
# Ablations of the learned attention weighting (fusion/attention/ablations.MD) on L0_bench (can be run on several GPUs at once).
# Same protocol as results/hard_scenario/fusion_bench: 500 demos, 101 epochs, last checkpoint, test sets (a) and (g)
# with 50 episodes during training (scripts/run_experiment.sh), then a re-evaluation with 200 episodes per test set
# and the sensor weights saved per step (--save_w) -> results/hard_scenario/fusion_ablations/n200/<run>/.
# Run from the repo root, Diffusion Policy at ../diffusion_policy (or DP_ROOT):
#
#   TIER=1 bash fusion/attention/run_ablations.sh                 # tier 1 on GPUs 0-3
#   TIER=2 GPUS="0 1" PER_GPU=2 bash fusion/attention/run_ablations.sh
#   ONLY="fusion_frozen fusion_anchor" bash fusion/attention/run_ablations.sh    # just these tags (both seeds)
#
# Settings: TIER (1 | 2 | 3 | all; default 1), GPUS ("0 1 2 3"), PER_GPU (runs at once per GPU, 1; A6000s fit 2),
# SEEDS ("42 43"), SEG_CKPT (segmentation checkpoint the encoder starts from; data/fusion/runs/mix_ws/best.pt),
# SEG_CKPT_STX (for P11; data/fusion_v2/runs/mix_ws_stx_supst/best.pt), N200 (1: the 200-episode re-evaluation),
# DRY (1: only print the run_experiment.sh command of every run, into the queue logs).
# Another level than L0_bench (fusion/attention/run_l0_vis.sh uses these): LEVEL (L0_bench), NDEMOS (500), GROUP (results
# folder, fusion_ablations), P0_SEEDS (seeds of the full model, "44": seeds 42 and 43 exist on L0_bench), ONLY_VARIANT
# (with ONLY: keep only this variant, e.g. rgbdt, since the tag `fusion` is both P0 and P10), BASE_EXTRA (hydra overrides
# added to every run), EGL_OF (EGL device of each GPU index, e.g. "3 2 0 1", where EGL numbers the GPUs differently from
# CUDA), TABLE (1: write the L0_bench results table at the end; default 1 on L0_bench, 0 elsewhere; N200 likewise). The settings of
# scripts/run_experiment.sh (TEST_SETS, NTEST, NEVAL, TOPK, ...) are passed on from the environment.
# Rerunning resumes unfinished runs and skips finished ones. Logs: data/runs/abl_queue_<gpu>_<slot>.log.
set -u
TIER=${TIER:-1}
GPUS=(${GPUS:-0 1 2 3})
PER_GPU=${PER_GPU:-1}
SEEDS=(${SEEDS:-42 43})
SEG_CKPT=${SEG_CKPT:-data/fusion/runs/mix_ws/best.pt}
SEG_CKPT_STX=${SEG_CKPT_STX:-data/fusion_v2/runs/mix_ws_stx_supst/best.pt}
LEVEL=${LEVEL:-L0_bench}
NDEMOS=${NDEMOS:-500}
N200=${N200:-$([ "$LEVEL" = L0_bench ] && echo 1 || echo 0)}     # the 200-episode re-evaluation is L0_bench's
P0_SEEDS=(${P0_SEEDS:-44})
BASE_EXTRA=${BASE_EXTRA:-}
EGL_OF=(${EGL_OF:-})
TABLE=${TABLE:-$([ "$LEVEL" = L0_bench ] && echo 1 || echo 0)}
export GROUP=${GROUP:-fusion_ablations} TRAIN=scripts/train_fusion_policy.sh EVAL_EXTRA=--save_w
mkdir -p data/runs

PRE="policy.obs_encoder.pretrained=$SEG_CKPT"
# tag | variant | tier | hydra overrides (no spaces inside one override)
EXPERIMENTS=(
  "fusion|rgbdt|1|$PRE"                                                                        # P0: 3rd seed only
  "fusion_frozen|rgbdt|1|$PRE policy.obs_encoder.freeze_segmenter=true"                         # P2
  "fusion_anchor|rgbdt|1|$PRE policy.obs_encoder.anchor=true policy.anchor_weight=1.0"          # P4
  "fusion_static|rgbdt|1|$PRE ++policy.obs_encoder.model_overrides={gate_mode:static}"           # P5
  "fusion_kp|rgbdt|1|$PRE policy.obs_encoder.readout=[avg,mask,centroid,area,max,gates,keypoint,ssm]"  # P9
  "fusion_scratch|rgbdt|2|++policy.obs_encoder.model_cfg={fusion_mode:weighted_sum,gate_norm:softmax}" # P1
  "fusion_gatesup|rgbdt|2|$PRE policy.gate_sup_weight=0.1"                                       # P6
  "fusion_aug|rgbdt|2|$PRE ++task.dataset.augment={thermal_contrast_p:0.5,depth_hole_p:0.0}"     # P7
  "fusion|rgbd|2|$PRE"                                                                           # P10
  "fusion_warm|rgbdt|3|$PRE policy.obs_encoder.freeze_steps=10000"                               # P3
  "fusion_global|rgbdt|3|$PRE policy.obs_encoder.readout=[avg,gates]"                            # P8
  "fusion_stx|rgbdt|3|policy.obs_encoder.pretrained=$SEG_CKPT_STX"                               # P11
)

jobs=()
for e in "${EXPERIMENTS[@]}"; do
  IFS="|" read -r tag variant tier extra <<< "$e"
  [ "$TIER" = all ] || [ "$TIER" = "$tier" ] || [ -n "${ONLY:-}" ] || continue
  [ -z "${ONLY:-}" ] || [[ " $ONLY " == *" $tag "* ]] || continue
  [ -z "${ONLY_VARIANT:-}" ] || [ "$variant" = "$ONLY_VARIANT" ] || continue
  if [ "$tag" = fusion ] && [ "$variant" = rgbdt ]; then
    for seed in "${P0_SEEDS[@]}"; do jobs+=("$tag|$variant|$seed|$extra"); done
    continue
  fi
  for seed in "${SEEDS[@]}"; do jobs+=("$tag|$variant|$seed|$extra"); done
done
[ ${#jobs[@]} -gt 0 ] || { echo "nothing to run for TIER=$TIER ONLY=${ONLY:-}"; exit 1; }
for f in "$SEG_CKPT"; do [ -f "$f" ] || echo "WARNING: $f not found (set SEG_CKPT)"; done

run_one() {  
  local gpu=$1 tag=$2 variant=$3 seed=$4 extra=$5
  local t=$tag; [ "$seed" = 42 ] || t=${tag}_seed$seed
  local name=${LEVEL}_${variant}_${NDEMOS}demos_$t
  [ -z "$BASE_EXTRA" ] || extra="$BASE_EXTRA $extra"
  [ ${#EGL_OF[@]} = 0 ] || export MUJOCO_EGL_DEVICE_ID=${EGL_OF[$gpu]}
  if [ "${DRY:-0}" = 1 ]; then
    echo "GPU=$gpu ${MUJOCO_EGL_DEVICE_ID:+MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID }TAG=$t EXTRA=\"training.seed=$seed $extra\" GROUP=$GROUP TRAIN=$TRAIN EVAL_EXTRA=$EVAL_EXTRA bash scripts/run_experiment.sh $LEVEL $variant $NDEMOS"
    return
  fi
  GPU=$gpu TAG=$t EXTRA="training.seed=$seed $extra" bash scripts/run_experiment.sh $LEVEL $variant $NDEMOS \
    || { echo "!!! FAILED: $name"; return; }
  local out=results/hard_scenario/fusion_ablations/n200/$name
  if [ "$N200" = 1 ] && [ ! -f "$out/g_heldout_all/eval_log.json" ]; then
    echo "[$(date +%H:%M)] $name: 200-episode evaluation -> $out"
    CUDA_VISIBLE_DEVICES=$gpu python policy/eval.py -c data/runs/$name/checkpoints/latest.ckpt -o $out \
      --level L0_bench --n_test 200 --n_envs 10 --n_vis 0 --test_set a_in_distribution --test_set g_heldout_all \
      --save_w > data/runs/$name.eval200.log 2>&1 || echo "!!! FAILED eval200: $name"
  fi
}

nq=$(( ${#GPUS[@]} * PER_GPU ))
echo "$(date): ${#jobs[@]} runs on GPUs ${GPUS[*]} ($PER_GPU per GPU):"
for j in "${!jobs[@]}"; do echo "  queue $(( j % nq )): ${jobs[$j]%%|*} $(echo "${jobs[$j]}" | cut -d'|' -f2,3)"; done
for q in $(seq 0 $((nq - 1))); do
  gpu=${GPUS[$(( q % ${#GPUS[@]} ))]}
  (
    for j in "${!jobs[@]}"; do
      [ $(( j % nq )) = $q ] || continue
      IFS="|" read -r tag variant seed extra <<< "${jobs[$j]}"
      run_one "$gpu" "$tag" "$variant" "$seed" "$extra"
    done
  ) > data/runs/abl_queue_${gpu}_$q.log 2>&1 &
done
wait
[ "${DRY:-0}" = 1 ] && { cat data/runs/abl_queue_*.log; exit 0; }
echo "$(date): all queues done"
grep -h "FAILED" data/runs/abl_queue_*.log || echo "no failed runs"
[ "$TABLE" = 1 ] || exit 0
python -m fusion.attention.ablation_table --spec fusion/attention/ablations.yaml --out results/hard_scenario/fusion_ablations/table