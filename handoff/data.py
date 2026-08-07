"""Fetch Project Sidewalk's released label crops and join them to the
disagreement targets.

This replaces the Street View Static API route in crops.py. Project Sidewalk
publishes the crops themselves, one dataset per label type, free and already
cut to the label. That removes the API key, the ~$450 of requests and the
unverified zoom->fov mapping in one step. `crops.py` is kept only as a fallback
for label types with no released dataset -- which, as it turns out, includes
the one this study most wanted.

FOUR THINGS FOUND IN THE RELEASE THAT CHANGE HOW IT MUST BE USED.

**1. There are no CurbRamp crops.** The `...-dataset-curbramp` repo is a
byte-identical copy of `...-dataset-crosswalk`: same splits, same classes, same
1,585 filenames. Every one of its rows that joins to the label export lands on
`label_type == Crosswalk`, 915 for 915. So the repo is mislabelled, not merely
duplicated, and CurbRamp -- the highest-volume label type and the call a city
actually budgets against -- has no imagery here. `REPOS` below deliberately
omits it.

**2. The class is the validation outcome, not the label type.** Folders are
`correct`/`incorrect`: whether reviewers upheld the crowdsourced label. That is
a better task than the class discrimination originally planned, because it *is*
the Sidewalk hand-off decision -- an inspector deciding whether to trust a
volunteer's label.

**3. The released splits leak.** 1,547 panoramas (24.4% of images) appear in
more than one of train/val/test. Two labels on one panorama share lighting,
camera and often the same stretch of pavement. The splits here are regenerated
grouped on `pano_id`; the released ones are recorded as `hf_split` for
reference and must not be used.

**4. The imaged subset is range-restricted on the thing being predicted.**
Assigning `correct`/`incorrect` needs consensus, so contested labels are
under-represented: 18.3% of the imaged subset has `split_adj > 0.5` against
30.2% of the full export. Disagreement survives -- roughly 3,360 contested
labels, sd 0.224 against 0.285 -- but it is truncated, and §Role B of DESIGN.md
explains why that makes a null result weaker evidence than a positive one.

Usage:
    python handoff/data.py --type crosswalk
    python handoff/data.py --type nocurbramp
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# `curbramp` is excluded on purpose; see the module docstring.
REPOS = {
    "nocurbramp": "projectsidewalk/sidewalk-validator-ai-dataset-nocurbramp",
    "obstacle": "projectsidewalk/sidewalk-validator-ai-dataset-obstacle",
    "surfaceproblem": "projectsidewalk/sidewalk-validator-ai-dataset-surfaceproblem",
    "crosswalk": "projectsidewalk/sidewalk-validator-ai-dataset-crosswalk",
}

# The label_type each repo actually contains, verified by joining every
# filename against the export. Checked at load time -- a silent change here
# would put the wrong pictures behind the right numbers.
EXPECTED_TYPE = {
    "nocurbramp": "NoCurbRamp",
    "obstacle": "Obstacle",
    "surfaceproblem": "SurfaceProblem",
    "crosswalk": "Crosswalk",
}

# The export calls Seattle `seattle`; the crop filenames call it `sea`.
CITY_ALIAS = {"sea": "seattle"}


def download(repo: str, cache: Path) -> Path:
    # The Xet backend rate-limits unauthenticated bulk pulls with a 429 partway
    # through; the classic HTTP path is slower but finishes. Set HF_TOKEN to
    # lift the limit and drop this.
    import os

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo, repo_type="dataset",
                                  local_dir=cache, allow_patterns=["*.webp"],
                                  tqdm_class=None))


def index(root: Path) -> pd.DataFrame:
    """One row per crop, keyed the same way targets.py keys labels."""
    rows = []
    for p in sorted(root.rglob("*.webp")):
        cls, hf_split = p.parent.name, p.parent.parent.name
        city, _, lid = p.stem.rpartition("_")
        rows.append({"path": str(p.relative_to(root)), "hf_split": hf_split,
                     "cls": cls, "city": CITY_ALIAS.get(city, city),
                     "label_id": lid})
    d = pd.DataFrame(rows)
    if d.empty:
        raise SystemExit(f"no .webp under {root}")
    d["uid"] = d.city + ":" + d.label_id
    return d


def split_by_pano(d: pd.DataFrame, seed: int = 0,
                  frac=(0.70, 0.15, 0.15)) -> pd.DataFrame:
    """Assign train/val/test whole panoramas at a time.

    Grouping on `pano_id` is the point; the released splits do not, and 24.4%
    of images sit on a panorama that straddles two of them.
    """
    rng = np.random.default_rng(seed)
    panos = d.pano_id.dropna().unique()
    panos = panos[rng.permutation(len(panos))]
    n_tr = int(len(panos) * frac[0])
    n_va = int(len(panos) * (frac[0] + frac[1]))
    where = {p: "train" for p in panos[:n_tr]}
    where.update({p: "val" for p in panos[n_tr:n_va]})
    where.update({p: "test" for p in panos[n_va:]})
    d = d.copy()
    d["split"] = d.pano_id.map(where)
    # a crop with no panorama id cannot be grouped safely, so it is not used
    d.loc[d.pano_id.isna(), "split"] = None
    return d


def build(kind: str, cache: Path, targets: Path) -> pd.DataFrame:
    root = download(REPOS[kind], cache / kind)
    d = index(root)
    print(f"{len(d):,} crops on disk")

    # `city` and `label_id` are already carried by the crop index and are both
    # encoded in `uid`; dropping them from the export side avoids a _x/_y
    # collision that would silently rename the columns the report reads.
    t = pd.read_parquet(targets).drop(columns=["city", "label_id"])
    m = d.merge(t, on="uid", how="left")
    joined = m.label_type.notna()
    print(f"joined to export: {joined.sum():,}/{len(m):,} ({joined.mean():.1%})")

    m = m[joined].copy()
    seen = m.label_type.value_counts()
    if len(seen) != 1 or seen.index[0] != EXPECTED_TYPE[kind]:
        raise SystemExit(
            f"{kind}: expected all {EXPECTED_TYPE[kind]}, got {seen.to_dict()}"
            " -- the release changed; re-verify before training")

    m["y"] = (m.cls == "incorrect").astype(int)  # 1 = reviewers overturned it
    m = split_by_pano(m)
    return m


def report(m: pd.DataFrame, kind: str) -> None:
    print(f"\n=== {kind}: {len(m):,} crops, {m.pano_id.nunique():,} panoramas, "
          f"{m.city.nunique()} cities ===")
    print(f"class balance: {m.y.mean():.1%} incorrect")

    g = m.groupby("split")
    print(f"\n  {'split':6s} {'n':>7} {'panos':>7} {'%inc':>6} "
          f"{'contested':>10} {'unsure':>7}")
    for s in ("train", "val", "test"):
        if s not in g.groups:
            continue
        x = g.get_group(s)
        print(f"  {s:6s} {len(x):7,} {x.pano_id.nunique():7,} "
              f"{x.y.mean():6.1%} {int((x.split_adj > 0.5).sum()):10,} "
              f"{int((x.unsure_adj > 0.15).sum()):7,}")

    leak = m.groupby("pano_id").split.nunique()
    print(f"\n  panos straddling our splits: {(leak > 1).sum()} (must be 0)")
    hf_leak = m.groupby("pano_id").hf_split.nunique()
    print(f"  panos straddling HF splits : {(hf_leak > 1).sum()} "
          f"({(m.pano_id.isin(hf_leak[hf_leak > 1].index)).mean():.1%} of crops)")

    print(f"\n  Role B evaluation mass")
    print(f"    contested (split_adj>0.5) : {int((m.split_adj > 0.5).sum()):,}")
    print(f"    unclear   (unsure_adj>.15): {int((m.unsure_adj > 0.15).sum()):,}")
    print(f"    split_adj  sd {m.split_adj.std():.3f}  "
          f"unsure_adj sd {m.unsure_adj.std():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="crosswalk", choices=sorted(REPOS))
    ap.add_argument("--cache", type=Path, default=ROOT / "handoff/data/hf")
    ap.add_argument("--targets", type=Path,
                    default=ROOT / "handoff/data/targets.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "handoff/data")
    a = ap.parse_args()

    m = build(a.type, a.cache, a.targets)
    a.out.mkdir(parents=True, exist_ok=True)
    dest = a.out / f"manifest_{a.type}.parquet"
    m.to_parquet(dest, index=False)
    report(m, a.type)
    print(f"\nwrote {dest} ({len(m):,} rows)")


if __name__ == "__main__":
    main()
