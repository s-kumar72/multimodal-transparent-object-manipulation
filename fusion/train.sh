#!/bin/bash
# Trains and tests the glass-fusion models. Full writeup in fusion/RUNNING.md.
#
#   bash fusion/train.sh                                   # all stages; finished work is skipped
#   STAGES="seg summary" RUNS="mix_ws" bash fusion/train.sh   # one run only
#
# Stages, in order:
#   setup    check packages, GPUs and rendering; check out the sim/hard-scenario branch into data/worktrees/sim_hard
#   sim      render sim training and test images with glass masks
#   real     check for the real datasets (DOWNLOAD_REAL=1 downloads them)
#   seg      train one model per run in RUNS (in parallel, one queue per GPU) and test it
#   summary  write the results table to results/fusion/summary.md
#   policy   optional, not run by default: train Diffusion Policy with a trained model as its image encoder
#
# Runs (RUNS). The name says the training data, plus an optional variant:
#   data:     sim | real | real2sim (real, then sim) | mix (sim + real)
#   variant:  ws (learned per-sensor weights, the main model) | static (fixed equal weights) | rgb (RGB only)
#             rgbd | rgbt | softmax | paper (Huo et al.'s exact fusion)
#   default:  mix_ws sim real real2sim mix mix_static mix_rgb   (without real data: sim_ws sim sim_static sim_rgb)
#
# Main settings (environment variables): RUNS, STAGES, GPUS (default: all), ITERS (20000), BATCH (64),
# DOWNLOAD_REAL (0), TRANSPARENT_DATA_ROOT (real data folder). The rest are listed in fusion/RUNNING.md.

set -euo pipefail
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$REPO"
STAGES=${STAGES:-"setup sim real seg summary"}
ITERS=${ITERS:-20000}
BATCH=${BATCH:-64}
SIM_LEVELS=${SIM_LEVELS:-"L1 L2 L3b L4b"}
N_TRAIN_EPS=${N_TRAIN_EPS:-400}
N_VAL_EPS=${N_VAL_EPS:-40}
N_TEST_EPS=${N_TEST_EPS:-50}
STRIDE=${STRIDE:-6}
WORKERS=${WORKERS:-$(nproc)}
DATA=${DATA:-data/fusion}
SIM_REF=${SIM_REF:-origin/sim/hard-scenario}
DOWNLOAD_REAL=${DOWNLOAD_REAL:-0}
INIT_THERMAL=${INIT_THERMAL:-}
# GPUs for the seg stage: "0 1 2 3" runs one queue of runs per GPU in parallel; default: every visible GPU
GPUS=${GPUS-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')}
REAL_ROOT=${TRANSPARENT_DATA_ROOT:-$HOME/datasets/transparent/unified}
export TRANSPARENT_DATA_ROOT=$REAL_ROOT
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-$MUJOCO_GL}
PY=${PYTHON:-python}

DATA=$(realpath -m "$DATA")          # absolute: some steps run from the sim checkout
mkdir -p "$DATA/logs"
LOG="$DATA/logs/train_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "== fusion/train.sh $(date)  repo $REPO  ($(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD))"
echo "   stages: $STAGES   log: $LOG"
has() { [[ " $STAGES " == *" $1 "* ]]; }

# ------------------------------------------------------------------ which sim to use
# The hard-scenario levels (L1-L4b) live on sim/hard-scenario until it is merged. Rather than merging it into this
# branch, the script checks it out next to the repo as a git worktree (data/worktrees/sim_hard, ignored by git) and
# only uses it to render frames. Once sim/hard-scenario is merged (sim/testsets.py exists here), this repo is used.
# Override with SIM_DIR=<a checkout> or SIM_REF=<branch / commit>.
resolve_sim() {
    if [ -n "${SIM_DIR:-}" ]; then return; fi
    if [ -f sim/testsets.py ]; then SIM_DIR=$REPO; return; fi
    SIM_DIR=$REPO/data/worktrees/sim_hard
    git fetch -q origin "${SIM_REF#origin/}" || echo "   (git fetch failed, using the local $SIM_REF)"
    if [ -d "$SIM_DIR" ]; then
        git -C "$SIM_DIR" checkout -q --detach "$SIM_REF"
    else
        git worktree add -q --detach "$SIM_DIR" "$SIM_REF"
    fi
    echo "   sim: worktree of $SIM_REF ($(git -C "$SIM_DIR" rev-parse --short HEAD)) at $SIM_DIR"
}
resolve_sim
[ -f "$SIM_DIR/sim/testsets.py" ] || { echo "   sim: $SIM_DIR has no hard-scenario levels, using main's scene"; SIM_LEVELS=main; }
sim_py() { (cd "$SIM_DIR" && PYTHONPATH="$REPO:$SIM_DIR:$SIM_DIR/sim:${PYTHONPATH:-}" $PY "$@"); }

# ------------------------------------------------------------------ setup
if has setup; then
    echo "== setup"
    $PY - <<'PY'
import importlib.util, sys
need = ["torch", "torchvision", "numpy", "PIL", "mujoco", "mink", "yaml", "pandas", "pyarrow"]
missing = [m for m in need if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"missing packages: {missing}\n  pip install -r sim/requirements.txt -r real_data/requirements.txt pyyaml")
import torch, torchvision
print(f"   python {sys.version.split()[0]}, torch {torch.__version__}, cuda {torch.cuda.is_available()}"
      + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
if not torch.cuda.is_available():
    print("   WARNING: no GPU, training will be very slow")
try:
    torchvision.models.resnet18(weights="IMAGENET1K_V1"); torchvision.models.resnet50(weights="IMAGENET1K_V1")
    print("   ImageNet weights cached")
except Exception as e:   # compute nodes without internet: run setup once on a login node first
    print(f"   WARNING: could not download ImageNet weights ({e}); runs need them unless --no_pretrained")
PY
    if ! sim_py -c "from env import GlassPickEnv; GlassPickEnv().reset(0)" 2>/dev/null; then
        echo "   MUJOCO_GL=$MUJOCO_GL cannot render here, trying osmesa"
        export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
        sim_py -c "from env import GlassPickEnv; GlassPickEnv().reset(0)" || { echo "   no working renderer"; exit 1; }
    fi
    echo "   rendering with MUJOCO_GL=$MUJOCO_GL"
fi

# ------------------------------------------------------------------ sim frames
if has sim; then
    echo "== sim: levels $SIM_LEVELS, $N_TRAIN_EPS train / $N_VAL_EPS val episodes per level, $N_TEST_EPS per test set"
    sim_py -m fusion.collect_sim --levels $SIM_LEVELS --n_train "$N_TRAIN_EPS" --n_val "$N_VAL_EPS" \
        --n_test "$N_TEST_EPS" --stride "$STRIDE" --workers "$WORKERS" --out "$DATA/sim"
fi

# ------------------------------------------------------------------ real data
real_list() {   # converted datasets in the manifest
    [ -f "$REAL_ROOT/manifest.parquet" ] || return 0
    $PY -c "import pandas as pd; print(' '.join(sorted(pd.read_parquet('$REAL_ROOT/manifest.parquet', columns=['dataset'])['dataset'].unique())))"
}
if has real; then
    echo "== real data in $REAL_ROOT"
    if [ ! -f "$REAL_ROOT/manifest.parquet" ] && [ "$DOWNLOAD_REAL" = 1 ]; then
        echo "   downloading + converting rgbt_glass mju_glass rgbd_gsd trobi (~9.5 GB download, ~18 GB converted)"
        $PY -m pip install -q -r real_data/requirements.txt
        bash real_data/download.sh rgbt_glass mju_glass rgbd_gsd trobi
        DS="rgbt_glass mju_glass rgbd_gsd trobi"
        [ -d "${TRANSPOSE_SRC:-$HOME/datasets/transparent/raw/transpose}" ] && DS="$DS transpose"
        $PY real_data/convert.py --datasets $DS --out "$REAL_ROOT" --workers "$WORKERS"
    fi
    if [ -f "$REAL_ROOT/manifest.parquet" ]; then
        echo "   have: $(real_list)"
    else
        echo "   none (runs that need it are skipped). Get it with DOWNLOAD_REAL=1 or real_data/README.md"
    fi
fi
REAL=$(real_list)
contains() { [[ " $1 " == *" $2 "* ]]; }
REAL_TRAIN=(); REAL_VAL=(); REAL_EVAL=()
for d in rgbt_glass transpose mju_glass rgbd_gsd; do      
    contains "$REAL" $d && REAL_TRAIN+=("real:$d:train") && REAL_VAL+=("real:$d:val")
done
for d in rgbt_glass transpose mju_glass trobi; do contains "$REAL" $d && REAL_EVAL+=("real:$d:test"); done
SIM_TRAIN=("sim:$DATA/sim/*_train.npz"); SIM_VAL=("sim:$DATA/sim/*_val.npz"); SIM_EVAL=("sim:$DATA/sim/*_test_*.npz")

# ------------------------------------------------------------------ segmentation runs
if [ -z "${RUNS:-}" ]; then
    if [ -n "$REAL" ]; then RUNS="mix_ws sim real real2sim mix mix_static mix_rgb"; else RUNS="sim_ws sim sim_static sim_rgb"; fi
fi
run_seg() {   # name
    local name=$1 data=${1%%_*} variant="" extra=() train=() val=() iters=$ITERS
    [[ $1 == *_* ]] && variant=${1#*_}
    if [ -f "$DATA/runs/$name/final.json" ]; then echo "== $name: done, skipping"; return; fi
    case $data in
        sim)      train=("${SIM_TRAIN[@]}"); val=("${SIM_VAL[@]}") ;;
        real)     train=("${REAL_TRAIN[@]}"); val=("${REAL_VAL[@]}") ;;
        mix)      train=("${SIM_TRAIN[@]}" "${REAL_TRAIN[@]}"); val=("${SIM_VAL[@]}" "${REAL_VAL[@]}") ;;
        real2sim) local base=real${variant:+_$variant}
                  run_seg "$base"
                  [ -f "$DATA/runs/$base/best.pt" ] || { echo "== $name: needs $base, skipping"; return; }
                  train=("${SIM_TRAIN[@]}"); val=("${SIM_VAL[@]}"); iters=$((ITERS / 2))
                  extra+=(--init "$DATA/runs/$base/best.pt" --lr 1e-4 --warmup 200) ;;
        paper)    contains "$REAL" rgbt_glass || { echo "== $name: needs rgbt_glass, skipping"; return; }
                  train=("real:rgbt_glass:train"); val=("real:rgbt_glass:val")
                  extra+=(--backbone resnet50 --size 224 --crop 208 --no_cues --gate_sup 0 --gate_logit_max 0 --batch_size 16 --eval "real:rgbt_glass:test")
                  [ "$variant" = rgb ] && extra+=(--modalities rgb) || extra+=(--modalities rgb thermal)
                  variant="" ;;
        *) echo "unknown run $name"; exit 1 ;;
    esac
    if [[ $data == real* || $data == mix ]] && [ -z "$REAL" ]; then echo "== $name: no real data, skipping"; return; fi
    case $variant in
        "") ;;
        static)  extra+=(--gate_mode static) ;;
        softmax) extra+=(--gate_norm softmax) ;;
        ws)      extra+=(--fusion_mode weighted_sum --gate_norm softmax) ;;
        paper)   extra+=(--no_cues --gate_sup 0 --gate_logit_max 0) ;;
        rgb)     extra+=(--modalities rgb) ;;
        rgbd)    extra+=(--modalities rgb depth) ;;
        rgbt)    extra+=(--modalities rgb thermal) ;;
        *) echo "unknown variant $variant in $name"; exit 1 ;;
    esac
    [ -n "$INIT_THERMAL" ] && [ "$data" != real2sim ] && extra+=(--init_thermal "$INIT_THERMAL")
    local evalargs=(--eval "${SIM_EVAL[@]}" "${REAL_EVAL[@]}")
    [ "$data" = paper ] && evalargs=()
    echo "== $name: train ${train[*]}"
    $PY -m fusion.train_seg --name "$name" --out_dir "$DATA/runs" --train "${train[@]}" --val "${val[@]}" \
        "${evalargs[@]}" --iters "$iters" --batch_size "$BATCH" --workers "$WORKERS" \
        --real_root "$REAL_ROOT" "${extra[@]}" ${SEG_EXTRA:-}
}
if has seg; then
    ls $DATA/sim/*_train.npz >/dev/null 2>&1 || { echo "no sim frames in $DATA/sim (stage sim)"; exit 1; }
    read -ra GPU_LIST <<< "$GPUS"
    if [ ${#GPU_LIST[@]} -le 1 ]; then
        [ ${#GPU_LIST[@]} -eq 1 ] && export CUDA_VISIBLE_DEVICES=${GPU_LIST[0]}
        echo "== seg runs: $RUNS ($ITERS iterations, batch $BATCH)"
        for r in $RUNS; do run_seg "$r"; done
    else
        NG=${#GPU_LIST[@]}
        WORKERS=$(( $(nproc) / NG > 2 ? $(nproc) / NG : 2 ))      
        if [ ${#REAL_TRAIN[@]} -gt 0 ]; then                       
            echo "== caching real data"
            $PY -c "import sys; from fusion.data import load_source
for s in sys.argv[1:]: load_source(s, 96, real_root='$REAL_ROOT')" "${REAL_TRAIN[@]}" "${REAL_VAL[@]}" "${REAL_EVAL[@]}"
        fi
        mapfile -t QUEUES < <($PY - "$NG" $RUNS <<'PY'
import sys
ng, runs = int(sys.argv[1]), sys.argv[2:]
chains = []
for r in runs:
    if r.startswith("real2sim"):
        base = "real" + r[len("real2sim"):]
        hit = next((c for c in chains if c[0] == base), None)
        if hit: hit.append(r); continue
    chains.append([r])
cost = lambda c: sum(0.5 if r.startswith("real2sim") else 1 for r in c) + (1 if c[0].startswith("real2sim") else 0)
load, queues = [0.0] * ng, [[] for _ in range(ng)]
for c in sorted(chains, key=cost, reverse=True):
    g = load.index(min(load)); queues[g] += c; load[g] += cost(c)
for q in queues: print(" ".join(r for r in runs if r in q))
PY
)
        echo "== seg runs on GPUs ${GPU_LIST[*]} ($ITERS iterations, batch $BATCH, $WORKERS loader workers each)"
        PIDS=()
        for i in "${!GPU_LIST[@]}"; do
            q=${QUEUES[$i]:-}
            [ -z "$q" ] && continue
            qlog="$DATA/logs/seg_gpu${GPU_LIST[$i]}_$(date +%Y%m%d_%H%M%S).log"
            echo "   GPU ${GPU_LIST[$i]}: $q   (log: $qlog)"
            ( export CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]}; for r in $q; do run_seg "$r"; done ) > "$qlog" 2>&1 &
            PIDS+=($!)
        done
        FAIL=0
        for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
        [ $FAIL = 0 ] || echo "   WARNING: a GPU queue failed; see data/fusion/logs/seg_gpu*.log (rerun to retry, finished runs are kept)"
    fi
fi

if has summary; then
    echo "== summary"
    $PY -m fusion.summarize --runs "$DATA/runs" --out results/fusion/summary.md
fi

# ------------------------------------------------------------------ policy (opt-in)
# Diffusion Policy on L4b with the fusion encoder initialized from a segmentation run, trained and evaluated the same
# way as the scripts/aws/sweep.sh on sim/hard-scenario, to compare with its L4b_rgbdt result.
# Needs Diffusion Policy (DP_ROOT, default ../diffusion_policy) and its packages (scripts/bridges2/setup.sh).
# Knobs: POLICY_INIT (run whose best.pt initializes the encoder, default mix_ws, the run whose gates are the proposal's
# w_m; pick the best one in results/fusion/summary.md; "none" = from scratch),
# POLICY_LEVEL (L4b), DEMOS (zarr; collected with the teacher if missing), N_DEMOS (200), POLICY_EPOCHS (101).
if has policy; then
    DP_ROOT=$(cd "${DP_ROOT:-../diffusion_policy}" && pwd)
    POLICY_INIT=${POLICY_INIT:-mix_ws}
    [ "$POLICY_INIT" = none ] || [ -f "$DATA/runs/$POLICY_INIT/best.pt" ] || { echo "no $DATA/runs/$POLICY_INIT/best.pt (stage seg, RUNS=$POLICY_INIT)"; exit 1; }
    POLICY_LEVEL=${POLICY_LEVEL:-L4b}
    N_DEMOS=${N_DEMOS:-200}
    DEMOS=$(realpath -m "${DEMOS:-$DATA/demos/sim_demos_${POLICY_LEVEL}_${N_DEMOS}.zarr}")
    POLICY_EPOCHS=${POLICY_EPOCHS:-101}
    [ -f "$SIM_DIR/sim/testsets.py" ] || { echo "policy stage needs the hard-scenario sim"; exit 1; }
    if [ ! -d "$DEMOS" ]; then
        echo "== policy: collecting $N_DEMOS $POLICY_LEVEL demos -> $DEMOS"
        mkdir -p "$(dirname "$DEMOS")"
        sim_py sim/collect_demos.py --config "sim/configs/$POLICY_LEVEL.yaml" --n "$N_DEMOS" --workers "$WORKERS" \
            --out "$DEMOS"
    fi
    INIT_ARG="policy.obs_encoder.pretrained=null"
    [ "$POLICY_INIT" = none ] || INIT_ARG="policy.obs_encoder.pretrained=$DATA/runs/$POLICY_INIT/best.pt"
    OUT=$REPO/data/outputs/fusion_policy/${POLICY_LEVEL}_rgbdt_fusion_${POLICY_INIT}
    echo "== policy: Diffusion Policy + fusion encoder ($POLICY_INIT) on $POLICY_LEVEL -> $OUT"
    (cd "$SIM_DIR" && PYTHONPATH="$SIM_DIR:$SIM_DIR/sim:$DP_ROOT:$REPO:${PYTHONPATH:-}" SDL_VIDEODRIVER=dummy \
        WANDB_MODE=${WANDB_MODE:-offline} $PY "$DP_ROOT/train.py" --config-dir="$REPO/fusion/config" \
        --config-name=train_glass_fusion_image task.zarr_path="$DEMOS" task.env_runner.level="$POLICY_LEVEL" \
        "$INIT_ARG" training.num_epochs="$POLICY_EPOCHS" task.env_runner.n_test=50 \
        exp_name="${POLICY_LEVEL}_rgbdt_fusion_${POLICY_INIT}" hydra.run.dir="$OUT" ${POLICY_EXTRA:-})
fi
echo "== done $(date)"