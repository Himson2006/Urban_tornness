#!/usr/bin/env bash
# Parity check + hero figures for the HICSS-style urban paper.
# The paper's Table 1 claim is "matches baseline accuracy while uniquely
# explaining", so ProtoPNet and the plain CNNs must be scored the same way on
# the same held-out folds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
GPU="${1:-0}"; mkdir -p results figures

echo "=== ProtoPNet (v3, val-selected) per fold ==="
for k in 0 1 2 3 4; do
  python src/evaluate.py --ckpt "runs/resnet34_fold${k}/best.pth" \
         --split kfold --fold "$k" --gpu "$GPU"
done 2>&1 | tee results/protopnet_v3_eval.out

echo "=== plain CNN per-architecture means ==="
for a in resnet18 resnet34 resnet50; do
  echo -n "  $a: "
  grep -h "TEST ped-AUROC" runs_plain/${a}_*.out 2>/dev/null | \
    awk '{s+=$NF;n++} END {if(n) printf "ped-AUROC %.4f over %d folds\n", s/n, n; else print "no runs"}'
done | tee results/plain_auroc.out

echo "=== hero figures ==="
python src/competing_prototypes.py --fold 0 --n 12 --gpu "$GPU" \
  2>&1 | tee results/handoff.out
