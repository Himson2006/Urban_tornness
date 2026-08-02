#!/usr/bin/env bash
# Download one PIE video set, extract crops, verify, then optionally delete the
# videos. This is the storage-disciplined loop: one set on disk at a time.
#
#   ./scripts/run_set.sh set05              # keep videos after extracting
#   ./scripts/run_set.sh set05 --delete     # delete videos IF checks pass
#
# Safe to re-run: wget resumes partial files, extraction resumes per clip.
set -euo pipefail

SET="${1:?usage: run_set.sh <setNN> [--delete]}"
DELETE="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIPS="$ROOT/PIE_clips"
LIST="$ROOT/data/pie_manifest/videos_needed.txt"

[[ -f "$LIST" ]] || { echo "missing $LIST -- run: python src/pie_manifest.py"; exit 1; }

echo "=== $SET: downloading ==="
mkdir -p "$CLIPS/$SET"
grep "/$SET/" "$LIST" | xargs -n1 -P4 wget -q --show-progress -c -P "$CLIPS/$SET"

echo "=== $SET: on disk ==="
du -sh "$CLIPS/$SET"

echo "=== $SET: extracting crops ==="
if python "$ROOT/src/pie_extract.py" --set "$SET" --clips "$CLIPS"; then
    echo "=== $SET: checks PASSED ==="
    echo "contact sheet: $ROOT/data/pie_crops/_spotcheck_$SET.jpg"
    if [[ "$DELETE" == "--delete" ]]; then
        echo "deleting videos for $SET"
        rm -rf "${CLIPS:?}/$SET"
    else
        echo "videos kept. free the space with:  rm -rf $CLIPS/$SET"
    fi
else
    echo "=== $SET: checks FAILED -- videos kept, nothing deleted ==="
    exit 1
fi
