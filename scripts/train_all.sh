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

echo "=== headline model, official PIE split (GPU ${GPUS[0]}) ==="
python src/train_pie.py --split official --arch "$ARCH" --epochs "$EPOCHS" \
       --gpu "${GPUS[0]}" 2>&1 | tee -a runs/official.out

echo "=== 5 grouped folds across GPUs: ${GPUS[*]} ==="
pids=()
for k in 0 1 2 3 4; do
    if [[ -f "runs/${ARCH}_fold${k}/DONE" ]]; then
        echo "  fold $k already complete, skipping"
        continue
    fi
    g="${GPUS[$(( k % ${#GPUS[@]} ))]}"
    echo "  fold $k -> GPU $g"
    python src/train_pie.py --split kfold --fold "$k" --arch "$ARCH" \
           --epochs "$EPOCHS" --gpu "$g" >> "runs/fold${k}.out" 2>&1 &
    last=$!
    pids+=("$last")
    # with one GPU, run strictly sequentially
    [[ ${#GPUS[@]} -eq 1 ]] && wait "$last"
done
[[ ${#pids[@]} -gt 0 ]] && wait

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
