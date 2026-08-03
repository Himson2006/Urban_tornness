#!/usr/bin/env bash
# Where am I? Run this after any disconnect to see what survived.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ARCH="${ARCH:-resnet34}"

echo "=== disk ==="
df -h . | tail -1
[[ -d PIE_clips ]] && du -sh PIE_clips/* 2>/dev/null

echo
echo "=== crop extraction ==="
for s in set01 set02 set03 set04 set05 set06; do
    if [[ -f "data/pie_crops/_meta_${s}.parquet" ]]; then
        n=$(python -c "import pandas as pd;print(f'{len(pd.read_parquet(\"data/pie_crops/_meta_${s}.parquet\")):,}')" 2>/dev/null || echo "?")
        done_v=$(python -c "import json;print(len(json.load(open('data/pie_crops/_state_${s}.json'))['done']))" 2>/dev/null || echo "?")
        echo "  $s: $n crops, $done_v clips done"
    else
        echo "  $s: not started"
    fi
done

echo
echo "=== training ==="
for r in runs/${ARCH}_*/; do
    [[ -d "$r" ]] || continue
    name=$(basename "$r")
    if [[ -f "$r/DONE" ]]; then
        acc=$(grep -oE "best (val|test) acc [0-9.]+" "$r/train.log" 2>/dev/null | tail -1)
        tst=$(cat "$r/final_test.txt" 2>/dev/null | tr "\n" " ")
        echo "  $name: COMPLETE ($acc) $tst"
    elif [[ -f "$r/ckpt.pth" ]]; then
        ep=$(grep -oE "checkpoint saved \(epoch [0-9]+" "$r/train.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+$")
        echo "  $name: in progress, last checkpoint epoch ${ep:-?}"
    else
        echo "  $name: started, no checkpoint yet"
    fi
done

echo
echo "=== tornness features ==="
ls -1 runs/${ARCH}_*/tornness_*.parquet 2>/dev/null | sed 's/^/  /' || echo "  none yet"

echo
echo "=== running processes ==="
pgrep -af "train_pie|tornness|pie_extract" | sed 's/^/  /' || echo "  nothing running"

echo
echo "To resume everything, just re-run the same command:"
echo "  ./scripts/train_all.sh 0"
