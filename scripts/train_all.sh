#!/usr/bin/env bash
# Train the headline model + 5 grouped folds, then extract tornness features.
#
#   ./scripts/train_all.sh              # single GPU, sequential (~12-24 h)
#   ./scripts/train_all.sh 0 1 2 3      # spread folds over GPUs 0-3
#
# RESUMABLE. Every stage is idempotent: finished runs are skipped via their DONE
# marker, an interrupted run restarts from its last per-epoch checkpoint, and
# tornness extraction skips folds whose parquet already exists. If the server
# kills you at any point, re-run the exact same command.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
GPUS=("$@"); [[ ${#GPUS[@]} -eq 0 ]] && GPUS=(0)
EPOCHS="${EPOCHS:-40}"
ARCH="${ARCH:-resnet34}"
mkdir -p runs

# The headline run has no sibling to share the machine with, so give it every
# GPU via DataParallel and scale the batch to match.
ALL_GPUS=$(IFS=,; echo "${GPUS[*]}")
BATCH=$(( 80 * ${#GPUS[@]} ))
echo "=== headline model, official PIE split (GPUs $ALL_GPUS, batch $BATCH) ==="
python src/train_pie.py --split official --arch "$ARCH" --epochs "$EPOCHS" \
       --gpu "$ALL_GPUS" --batch "$BATCH" ${TF32:+--tf32} 2>&1 | tee -a runs/official.out

echo "=== 5 grouped folds across GPUs: ${GPUS[*]} ==="
# One fold per GPU at a time. With 5 folds and 4 GPUs the 5th waits for a free
# slot rather than doubling up and OOMing (or halving throughput) on GPU 0.
# indexed (not associative) array: bash 3.2 compatible, and `set -e` would
# abort the whole script if `declare -A` were unsupported
slot_pid=()
for i in "${!GPUS[@]}"; do slot_pid[$i]=""; done

for k in 0 1 2 3 4; do
    if [[ -f "runs/${ARCH}_fold${k}/DONE" ]]; then
        echo "  fold $k already complete, skipping"
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
    echo "  fold $k -> GPU $g"
    python src/train_pie.py --split kfold --fold "$k" --arch "$ARCH" \
           --epochs "$EPOCHS" --gpu "$g" ${TF32:+--tf32} \
           >> "runs/fold${k}.out" 2>&1 &
    slot_pid[$slot]=$!
    sleep 5   # stagger startup so they don't all hit the crop store at once
done
wait

echo "=== tornness features, each fold scored on its held-out pedestrians ==="
for k in 0 1 2 3 4; do
    out="runs/${ARCH}_fold${k}/tornness_fold${k}.parquet"
    if [[ -f "$out" ]]; then
        echo "  fold $k features already extracted, skipping"
        continue
    fi
    python src/tornness.py --ckpt "runs/${ARCH}_fold${k}/best.pth" \
           --split kfold --fold "$k" --gpu "${GPUS[0]}" 2>&1 | tee -a "runs/tornness${k}.out"
done

echo
echo "ALL DONE."
echo "Per-fold features: runs/${ARCH}_fold*/tornness_fold*.parquet"
echo "Prototype exemplars: runs/${ARCH}_fold*/img/"
