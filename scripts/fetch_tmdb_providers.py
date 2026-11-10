"""Fetch TMDB watch providers for every MovieLens movie that has a tmdbId.

    export TMDB_TOKEN=...   # TMDB "API Read Access Token" (free account)
    python scripts/fetch_tmdb_providers.py --ml-dir data/ml-latest-small --limit 3000

Writes data/providers.json and caches raw responses in data/tmdb_cache/ so
re-runs are free. Start with --limit to stay polite with the API.
"""
import argparse
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wpo.data.tmdb import fetch_providers  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ml-dir", default=str(ROOT / "data" / "ml-latest-small"))
ap.add_argument("--out", default=str(ROOT / "data" / "providers.json"))
ap.add_argument("--region", default="US")
ap.add_argument("--limit", type=int, default=3000, help="most-rated movies first")
a = ap.parse_args()

links = pd.read_csv(pathlib.Path(a.ml_dir) / "links.csv").dropna(subset=["tmdbId"])
pop = pd.read_csv(pathlib.Path(a.ml_dir) / "ratings.csv").groupby("movieId").size()
links["pop"] = links["movieId"].map(pop).fillna(0)
ids = links.sort_values("pop", ascending=False)["tmdbId"].astype(int).head(a.limit)
res = fetch_providers(ids, a.out, region=a.region, cache_dir=ROOT / "data" / "tmdb_cache")
n_any = sum(1 for v in res.values() if any(v.values()))
print(f"wrote {a.out}: {len(res)} titles, {n_any} with at least one provider in {a.region}")
