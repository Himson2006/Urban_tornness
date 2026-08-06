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


def fetch_one(rel: str, dest: Path, tries: int = 5) -> bool:
    """Fetch with backoff. The Hub returns 429 under naive parallel fetching."""
    import random
    import time
    out = dest / rel.replace("/", "__")
    if out.exists() and out.stat().st_size > 10:
        return True
    for k in range(tries):
        try:
            with urllib.request.urlopen(RAW + rel, timeout=60) as r:
                out.write_bytes(r.read())
            return True
        except urllib.error.HTTPError as e:
            if e.code != 429:
                return False
            time.sleep((2 ** k) + random.random())
        except Exception:
            time.sleep(1 + random.random())
    return False


def fetch_via_hub(dest: Path) -> int:
    """Preferred path: snapshot_download handles throttling and resumption."""
    from huggingface_hub import snapshot_download
    local = snapshot_download(
        repo_id=REPO, repo_type="dataset",
        allow_patterns=["*/labels/*post_disaster.json"],
        max_workers=4)
    n = 0
    for src in Path(local).rglob("*post_disaster.json"):
        rel = src.relative_to(local).as_posix()
        out = dest / rel.replace("/", "__")
        if not out.exists():
            out.write_bytes(src.read_bytes())
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "xbd/data/labels")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    files = list_label_files()
    print(f"{len(files):,} post-disaster label files in {REPO}")

    try:
        n = fetch_via_hub(a.out)
        print(f"  snapshot_download: {n:,} label files")
    except Exception as e:
        print(f"  snapshot_download unavailable ({type(e).__name__}); "
              f"falling back to direct fetch with backoff")
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(lambda f: fetch_one(f, a.out), files))

    have = len(list(a.out.glob("*.json")))
    print(f"  {have:,}/{len(files):,} on disk "
          f"({sum(p.stat().st_size for p in a.out.glob('*.json'))/1e6:.0f} MB)")
    if have < len(files):
        print("  re-run to retry the remainder; existing files are reused")


if __name__ == "__main__":
    main()
