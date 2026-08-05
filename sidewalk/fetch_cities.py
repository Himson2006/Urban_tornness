"""Download Project Sidewalk labels for every live city instance.

Each city runs its own server at sidewalk-<city>.cs.washington.edu. The v3
rawLabels endpoint returns one row per label with the full validation
breakdown -- agree_count, disagree_count, unsure_count -- plus the labeller's
id, a free-text description, severity, timestamps and geography.

That validation breakdown is the whole point. Reviewers can fail to agree in
two different ways: they split into camps (agree vs disagree) or they are
individually unsure ("not sure"). Almost no crowdsourcing dataset records the
second separately, which is why this distinction has been hard to study.

Usage:
    python sidewalk/fetch_cities.py                 # all known cities
    python sidewalk/fetch_cities.py --cities seattle columbus
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Verified live 2026-08. Others exist; --cities lets you add them.
CITIES = ["seattle", "columbus", "chicago", "cdmx", "spgg", "newberg",
          "oradell", "pittsburgh", "amsterdam", "zurich", "taipei", "burnaby"]

URL = ("https://sidewalk-{city}.cs.washington.edu/v3/api/rawLabels"
       "?filetype=csv")


def fetch(city: str, dest: Path, timeout: int = 300) -> bool:
    out = dest / f"{city}.csv"
    if out.exists() and out.stat().st_size > 1000:
        print(f"  {city:12s} cached ({out.stat().st_size/1e6:.1f} MB)")
        return True
    try:
        t0 = time.time()
        req = urllib.request.Request(
            URL.format(city=city),
            headers={"User-Agent": "research-script/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        if len(data) < 1000:
            print(f"  {city:12s} FAILED (response only {len(data)} bytes)")
            return False
        out.write_bytes(data)
        print(f"  {city:12s} {len(data)/1e6:6.1f} MB  ({time.time()-t0:.0f}s)")
        return True
    except Exception as e:
        print(f"  {city:12s} FAILED ({type(e).__name__}: {e})")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=CITIES)
    ap.add_argument("--out", type=Path, default=ROOT / "sidewalk/data/raw")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    print(f"downloading {len(a.cities)} cities -> {a.out}")
    ok = [c for c in a.cities if fetch(c, a.out)]
    total = sum(f.stat().st_size for f in a.out.glob("*.csv"))
    print(f"\n{len(ok)}/{len(a.cities)} cities, {total/1e6:.0f} MB on disk")
    if len(ok) < len(a.cities):
        print("re-run to retry failures; existing files are reused")


if __name__ == "__main__":
    main()
