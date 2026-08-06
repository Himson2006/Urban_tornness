"""Step A for xBD: pull the label JSONs only, no imagery.

Same discipline as the PIE and Project Sidewalk work -- establish what the
population looks like before committing to a download measured in tens of
gigabytes. Each post-disaster label file carries every building in the scene
with its damage class, its polygon in both pixel and geographic coordinates,
and the capture conditions of the image it came from.

That is enough to answer the questions that decide whether the study is worth
running: how many buildings sit on the contested boundary between adjacent
damage classes, how large a building is in pixels (the failure that sank the
pedestrian work), and how the classes distribute across disasters.

Source is a public Hugging Face mirror of xBD. The canonical release is at
xview2.org and requires registration; verify licence terms against that before
publishing.

Usage:
    python xbd/fetch_labels.py
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "aryananand/xBD"
API = f"https://huggingface.co/api/datasets/{REPO}"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main/"


def list_label_files() -> list[str]:
    with urllib.request.urlopen(API, timeout=60) as r:
        meta = json.load(r)
    files = [s["rfilename"] for s in meta.get("siblings", [])]
    # only post-disaster files carry damage classes; pre-disaster are all
    # "no-damage" by construction and add nothing to the label population
    return sorted(f for f in files
                  if f.endswith(".json") and "post_disaster" in f)


def fetch_one(rel: str, dest: Path) -> bool:
    out = dest / rel.replace("/", "__")
    if out.exists() and out.stat().st_size > 10:
        return True
    try:
        with urllib.request.urlopen(RAW + rel, timeout=60) as r:
            out.write_bytes(r.read())
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "xbd/data/labels")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    files = list_label_files()
    print(f"{len(files):,} post-disaster label files in {REPO}")

    with ThreadPoolExecutor(a.workers) as ex:
        ok = list(ex.map(lambda f: fetch_one(f, a.out), files))
    have = len(list(a.out.glob("*.json")))
    print(f"  {sum(ok):,}/{len(files):,} fetched, {have:,} on disk "
          f"({sum(p.stat().st_size for p in a.out.glob('*.json'))/1e6:.0f} MB)")
    if sum(ok) < len(files):
        print("  re-run to retry failures; existing files are reused")


if __name__ == "__main__":
    main()
