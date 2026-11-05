#!/bin/bash
# Difficulty sweep on one GPU: for each level x modality, train Diffusion Policy on 200 demos, then evaluate the last
# EMA checkpoint on test sets (a) in-distribution and (g) everything held out (50 episodes each).
#   bash scripts/aws/sweep.sh "L4b L0" "rgb rgbd rgbdt"
# Training outputs and checkpoints stay on the ephemeral NVMe disk ($WORK/sweep/<level>_<variant>); after every run the
# training log, the evaluation logs and per-episode records are copied to results/hard_scenario/sweep/ and pushed.
# A finished run (eval_log.json present for both test sets) is skipped, so the script can be rerun after a reboot
# (after bash scripts/aws/setup_nvme.sh).
set -e
LEVELS=${1:-"L4b L0"}
VARIANTS=${2:-"rgb rgbd rgbdt"}
EPOCHS=${EPOCHS:-101}       # 100 epochs + the evaluation at epoch 100 (Diffusion Policy evaluates when epoch % 25 == 0)
NTEST=${NTEST:-50}
NENVS=${NENVS:-10}          # envs rendered at once; more do not fit next to the model on a 16 GB T4
WORK=/opt/dlami/nvme/work
RES=results/hard_scenario/sweep
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
        mkdir -p $out
        echo "$name: training ($(date))"
        policy/train.sh $v $zarr training.num_epochs=$EPOCHS task.env_runner.n_test=$NTEST \
            task.env_runner.n_envs=$NENVS task.env_runner.level=$level \
            checkpoint.topk.k=1 hydra.run.dir=$run > $run.log 2>&1 || { echo "$name: training failed, see $run.log"; exit 1; }
        cp $run/logs.json.txt $out/train_logs.json.txt
        grep -a "test/mean_score" $run.log | tr -d '\r' | grep -o "test/mean_score.*" > $out/train_scores.txt || true
        echo "$name: evaluating ($(date))"
        python policy/eval.py -c $run/checkpoints/latest.ckpt -o $out --level $level --n_test $NTEST --n_envs $NENVS \
            --n_vis 0 --test_set a_in_distribution --test_set g_heldout_all > $run.eval.log 2>&1 \
            || { echo "$name: evaluation failed, see $run.eval.log"; exit 1; }
        git add $out
        git commit -qm "Sweep result: $name ($EPOCHS epochs, 200 demos)"
        git push -q || echo "WARNING: push failed for $name"
        echo "$name: $(grep -h '"test/mean_score"' $out/*/eval_log.json | tr -d ' ,')"
    done
done
