#!/usr/bin/env bash
# Train the headline model + 5 grouped folds, then extract tornness features.
#
#   ./scripts/train_all.sh                    # single GPU, sequential
#   ./scripts/train_all.sh 0 1 2 3 4 5        # 6 jobs, one GPU each, one wave
#   TF32=1 WORKERS=16 ./scripts/train_all.sh 0 1 2 3 4 5
#
# There are exactly 6 training jobs (official + folds 0-4). Given >=6 GPUs they
# all run concurrently on one GPU each, with identical hyperparameters -- which
# matters: sharding one run across GPUs would force a different batch size for
# the headline model than for the folds, and batch size is an optimisation
# variable, not just a speed knob.
#
# RESUMABLE. Finished runs are skipped via their DONE marker, interrupted runs
# restart from their last per-epoch checkpoint, and tornness extraction skips
# folds whose parquet exists. After any interruption, re-run the same command.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
GPUS=("$@"); [[ ${#GPUS[@]} -eq 0 ]] && GPUS=(0)
EPOCHS="${EPOCHS:-40}"
ARCH="${ARCH:-resnet34}"
BATCH="${BATCH:-80}"
# Total dataloader workers should not exceed core count: concurrent jobs each
# spawn their own, and oversubscribing the CPU makes everything slower.
NCORE=$(nproc 2>/dev/null || echo 8)
NJOB=6
WORKERS="${WORKERS:-$(( NCORE / NJOB > 16 ? 16 : NCORE / NJOB ))}"
[[ $WORKERS -lt 2 ]] && WORKERS=2
mkdir -p runs

echo "GPUs: ${GPUS[*]} | cores: $NCORE | workers/job: $WORKERS | batch: $BATCH"
echo "epochs: $EPOCHS | arch: $ARCH | tf32: ${TF32:-off}"
echo

# job list: "official" plus the five folds
JOBS=(official 0 1 2 3 4)

run_dir_for() {
    [[ "$1" == "official" ]] && echo "runs/${ARCH}_official" || echo "runs/${ARCH}_fold$1"
}

launch() {   # $1 = job, $2 = gpu
    local job="$1" g="$2" args
    if [[ "$job" == "official" ]]; then
        args=(--split official)
    else
        args=(--split kfold --fold "$job")
    fi
    # -u: unbuffered stdout, so runs/<job>.out stays live and survives a kill
    python -u src/train_pie.py "${args[@]}" --arch "$ARCH" --epochs "$EPOCHS" \
           --batch "$BATCH" --workers "$WORKERS" --gpu "$g" ${TF32:+--tf32} \
           >> "runs/${job}.out" 2>&1 &
}

echo "=== training: ${#JOBS[@]} jobs across ${#GPUS[@]} GPUs ==="
slot_pid=()
for i in "${!GPUS[@]}"; do slot_pid[$i]=""; done

for job in "${JOBS[@]}"; do
    d=$(run_dir_for "$job")
    if [[ -f "$d/DONE" ]]; then
        echo "  $job already complete, skipping"
        continue
    fi
    slot=-1
    while [[ $slot -lt 0 ]]; do
        for i in "${!GPUS[@]}"; do
            pid="${slot_pid[$i]}"
            if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                slot=$i; break
            fi
        done
        [[ $slot -lt 0 ]] && sleep 20
    done
    g="${GPUS[$slot]}"
    echo "  $job -> GPU $g  (log: runs/${job}.out)"
    launch "$job" "$g"
    slot_pid[$slot]=$!
    sleep 5   # stagger startup so they don't all hit the crop store at once
done
wait

echo
echo "=== tornness features, each fold scored on its held-out pedestrians ==="
for k in 0 1 2 3 4; do
    out="runs/${ARCH}_fold${k}/tornness_fold${k}.parquet"
    if [[ -f "$out" ]]; then
        echo "  fold $k features already extracted, skipping"
        continue
    fi
    g="${GPUS[$(( k % ${#GPUS[@]} ))]}"
    python src/tornness.py --ckpt "runs/${ARCH}_fold${k}/best.pth" \
           --split kfold --fold "$k" --workers "$WORKERS" --gpu "$g" \
           2>&1 | tee -a "runs/tornness${k}.out"
done

echo
echo "ALL DONE."
echo "Per-fold features:   runs/${ARCH}_fold*/tornness_fold*.parquet"
echo "Prototype exemplars: runs/${ARCH}_fold*/img/"
