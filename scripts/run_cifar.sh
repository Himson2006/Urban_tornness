#!/usr/bin/env bash
# CIFAR-10H arm: train, extract 10-class tornness, run the shape experiment.
# Resumable -- re-run the same command after any interruption.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
GPU="${1:-0}"
EPOCHS="${EPOCHS:-30}"
mkdir -p runs_cifar

python -c "import pyarrow" || { echo "pyarrow missing -- wrong conda env?"; exit 1; }

echo "=== 1/4 human label distributions ==="
[[ -f data/cifar10h/human_labels.parquet ]] || python src/cifar10h_data.py

echo "=== 2/4 train (GPU $GPU) ==="
python -u src/cifar_train.py --gpu "$GPU" --epochs "$EPOCHS" --tf32 \
       2>&1 | tee -a runs_cifar/train.out

echo "=== 3/4 tornness on the CIFAR-10H test images ==="
python -u src/cifar_tornness.py --gpu "$GPU" 2>&1 | tee -a runs_cifar/tornness.out

echo "=== 4/4 shape experiment ==="
python src/exp1_cifar.py 2>&1 | tee -a runs_cifar/exp1.out
