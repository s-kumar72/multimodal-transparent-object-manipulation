#!/bin/bash
# Difficulty sweep on one GPU: for each level x modality, train Diffusion Policy on 200 demos, then evaluate the last
# EMA checkpoint on test sets (a) in-distribution and (g) everything held out (50 episodes each).
#   bash scripts/aws/sweep.sh "L4b" "rgbdt"
# Training outputs and checkpoints live on the ephemeral NVMe disk ($WORK/sweep/<level>_<variant>), which a reboot
# wipes. So while a run trains, every new latest.ckpt (saved every CKPT_EVERY epochs) is copied with the training log
# to $CKPT_DIR/<run>/ on the root disk; rerunning the script after a reboot (and scripts/aws/setup_nvme.sh) restores
# them and resumes training from there. After a run, its training log, score curve, evaluation logs and per-episode
# records go to $RES/<run>/ and are committed and pushed; the root-disk copy of the checkpoint is then deleted.
# A finished run (eval_log.json present for both test sets) is skipped.
#
# Knobs (environment): EPOCHS (101 = 100 + the evaluation at epoch 100; Diffusion Policy evaluates when epoch % 25 == 0),
# NTEST, NENVS, CKPT_EVERY, ROLLOUT_EVERY, EXTRA (more hydra overrides), RES, CKPT_DIR, COMMIT (0 = do not commit).
# CKPT_EVERY must be a multiple of ROLLOUT_EVERY: the top-k checkpoint manager needs a test score at every save.
set -e
LEVELS=${1:-"L4b L0"}
VARIANTS=${2:-"rgb rgbd rgbdt"}
EPOCHS=${EPOCHS:-101}
NTEST=${NTEST:-50}
NENVS=${NENVS:-10}          # envs rendered at once; more do not fit next to the model on a 16 GB T4
CKPT_EVERY=${CKPT_EVERY:-25}
ROLLOUT_EVERY=${ROLLOUT_EVERY:-25}
WORK=/opt/dlami/nvme/work
RES=${RES:-results/hard_scenario/sweep}
CKPT_DIR=${CKPT_DIR:-results/checkpoint/sweep}
COMMIT=${COMMIT:-1}

complete() {  # a torch checkpoint is a zip whose directory is written last, so a partial file does not open
    python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).close()" $1 2>/dev/null
}

backup() {  # run_dir backup_dir: copy latest.ckpt + logs once the (threaded) save has finished writing it
    # Diffusion Policy writes latest.ckpt and the top-k checkpoint from Python threads that share the GIL with
    # training, so a half-written file can sit unchanged for a while: check the zip, not the size or the mtime.
    local src=$1/checkpoints/latest.ckpt dst=$2
    [ -f $src ] || return 0
    [ -f $dst/latest.ckpt ] && [ ! $src -nt $dst/latest.ckpt ] && return 0
    complete $src || return 0
    local size=$(stat -c %s $src) mtime=$(stat -c %Y $src)
    mkdir -p $dst
    rm -f $dst/latest.ckpt $dst/latest.ckpt.part   # the root disk has room for one copy only
    cp -p $src $dst/latest.ckpt.part && cp $1/logs.json.txt $dst/logs.json.txt
    # keep the copy only if the source did not change while copying and the copy is a complete zip
    if [ "$(stat -c %s $src)" = "$size" ] && [ "$(stat -c %Y $src)" = "$mtime" ] && complete $dst/latest.ckpt.part; then
        mv $dst/latest.ckpt.part $dst/latest.ckpt
    else
        rm -f $dst/latest.ckpt.part; return 0      # retried at the next poll
    fi
    # checkpoints are saved at evaluation epochs, so the last evaluated epoch in the log is the checkpoint's
    echo "  backed up the epoch $(grep -a '"test/mean_score"' $dst/logs.json.txt | tail -1 | grep -o '"epoch": [0-9]*' | grep -o '[0-9]*$') checkpoint to $dst ($(date +%H:%M))"
}

curve() {  # logs.json.txt -> curve.json: the evaluation scores, one entry per evaluated epoch (last one wins on resume)
    python - "$1" "$2" <<'PY'
import json, sys
rows = {}
for line in open(sys.argv[1]):
    d = json.loads(line)
    if "test/mean_score" in d:
        rows[d["epoch"]] = {k: v for k, v in d.items() if k.startswith(("test/", "train/")) or k in ("epoch", "global_step", "val_loss", "train_loss")}
json.dump([rows[e] for e in sorted(rows)], open(sys.argv[2], "w"), indent=1)
print("  curve:", ", ".join(f"ep{e} {rows[e]['test/mean_score']:.2f}" for e in sorted(rows)))
PY
}

for level in $LEVELS; do
    zarr=results/data/sim_demos_${level}_200.zarr
    [ -d $zarr ] || { echo "missing $zarr"; exit 1; }
    for v in $VARIANTS; do
        name=${level}_$v
        out=$RES/$name
        if [ -f $out/a_in_distribution/eval_log.json ] && [ -f $out/g_heldout_all/eval_log.json ]; then
            echo "$name: done, skipping"; continue
        fi
        run=$WORK/sweep/$name
        mkdir -p $out $WORK/sweep
        if [ ! -f $run/checkpoints/latest.ckpt ] && [ -f $CKPT_DIR/$name/latest.ckpt ]; then
            echo "$name: restoring the checkpoint from $CKPT_DIR/$name (NVMe was wiped)"
            mkdir -p $run/checkpoints
            cp -p $CKPT_DIR/$name/latest.ckpt $run/checkpoints/latest.ckpt     # -p: same mtime, so no re-backup
            cp -p $CKPT_DIR/$name/logs.json.txt $run/logs.json.txt
        fi
        echo "$name: training ($(date))"
        policy/train.sh $v $zarr training.num_epochs=$EPOCHS task.env_runner.n_test=$NTEST \
            task.env_runner.n_envs=$NENVS task.env_runner.level=$level checkpoint.topk.k=1 \
            training.checkpoint_every=$CKPT_EVERY training.rollout_every=$ROLLOUT_EVERY \
            hydra.run.dir=$run ${EXTRA:-} >> $run.log 2>&1 &
        pid=$!
        while kill -0 $pid 2>/dev/null; do
            backup $run $CKPT_DIR/$name
            sleep 40
        done
        wait $pid || { echo "$name: training failed, see $run.log"; exit 1; }
        cp $run/logs.json.txt $out/train_logs.json.txt
        curve $run/logs.json.txt $out/curve.json
        echo "$name: evaluating ($(date))"
        python policy/eval.py -c $run/checkpoints/latest.ckpt -o $out --level $level --n_test $NTEST --n_envs $NENVS \
            --n_vis 0 --test_set a_in_distribution --test_set g_heldout_all > $run.eval.log 2>&1 \
            || { echo "$name: evaluation failed, see $run.eval.log"; exit 1; }
        if [ "$COMMIT" = 1 ]; then
            git add $out
            git commit -qm "Sweep result: $name ($EPOCHS epochs, 200 demos)"
            git push -q || echo "WARNING: push failed for $name"
        fi
        rm -rf $CKPT_DIR/$name     # results are in git; the checkpoint itself stays on NVMe
        echo "$name: $(grep -h '"test/mean_score"' $out/*/eval_log.json | tr -d ' ,')"
    done
done
