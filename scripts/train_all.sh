#!/usr/bin/env bash
# Train the headline model + 5 grouped folds, then extract tornness features.
#
#   ./scripts/train_all.sh              # single GPU, sequential (~12-24 h)
#   ./scripts/train_all.sh 0 1 2 3      # spread folds over GPUs 0-3
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
GPUS=("$@"); [[ ${#GPUS[@]} -eq 0 ]] && GPUS=(0)
EPOCHS="${EPOCHS:-40}"
ARCH="${ARCH:-resnet34}"
mkdir -p runs

echo "=== headline model, official PIE split (GPU ${GPUS[0]}) ==="
python src/train_pie.py --split official --arch "$ARCH" --epochs "$EPOCHS" --gpu "${GPUS[0]}"

echo "=== 5 grouped folds across GPUs: ${GPUS[*]} ==="
pids=()
for k in 0 1 2 3 4; do
    g="${GPUS[$(( k % ${#GPUS[@]} ))]}"
    echo "  fold $k -> GPU $g"
    python src/train_pie.py --split kfold --fold "$k" --arch "$ARCH" \
           --epochs "$EPOCHS" --gpu "$g" > "runs/fold${k}.out" 2>&1 &
    last=$!
    pids+=("$last")
    # with one GPU, run strictly sequentially
    [[ ${#GPUS[@]} -eq 1 ]] && wait "$last"
done
wait

echo "=== tornness features, each fold scored on its held-out pedestrians ==="
for k in 0 1 2 3 4; do
    python src/tornness.py --ckpt "runs/${ARCH}_fold${k}/best.pth" \
           --split kfold --fold "$k" --gpu "${GPUS[0]}"
done

echo
echo "DONE. Per-fold features: runs/${ARCH}_fold*/tornness_fold*.parquet"
echo "Prototype exemplars:     runs/${ARCH}_fold*/img/"
