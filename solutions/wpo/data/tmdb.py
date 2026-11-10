"""Fetch stream / rent / buy availability per movie from TMDB's watch-providers endpoint.

    GET https://api.themoviedb.org/3/movie/{tmdb_id}/watch/providers
    Authorization: Bearer <your TMDB API Read Access Token>

TMDB's provider data comes from JustWatch and is per region. Attribution is
required in anything you publish: "This product uses the TMDB API but is not
endorsed or certified by TMDB." Check TMDB's current terms and rate limits.

Output format (providers.json), keyed by tmdbId as a string:
    {"862": {"flatrate": ["Disney Plus"], "rent": ["Apple TV"], "buy": ["Apple TV"]}, ...}
"""
from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Iterable

API = "https://api.themoviedb.org/3/movie/{id}/watch/providers"


def fetch_providers(tmdb_ids: Iterable[int], out_path, region: str = "US",
                    token: str | None = None, cache_dir="data/tmdb_cache", sleep: float = 0.05) -> dict:
    import requests  # imported here so the rest of the package works without it

    token = token or os.environ.get("TMDB_TOKEN")
    if not token:
        raise RuntimeError("set TMDB_TOKEN to your TMDB API Read Access Token")
    headers = {"Authorization": f"Bearer {token}", "accept": "application/json"}
    cache = pathlib.Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    result = {}
    for i, tid in enumerate(tmdb_ids):
        tid = int(tid)
        f = cache / f"{tid}.json"
        if f.exists():
            raw = json.loads(f.read_text())
        else:
            resp = requests.get(API.format(id=tid), headers=headers, timeout=15)
            if resp.status_code == 429:           # rate limited: back off and retry once
                time.sleep(2.0)
                resp = requests.get(API.format(id=tid), headers=headers, timeout=15)
            if resp.status_code == 404:
                raw = {}
            else:
                resp.raise_for_status()
                raw = resp.json()
            f.write_text(json.dumps(raw))
            time.sleep(sleep)
        reg = raw.get("results", {}).get(region, {})
        result[str(tid)] = {k: [p["provider_name"] for p in reg.get(k, [])] for k in ("flatrate", "rent", "buy")}
        if (i + 1) % 200 == 0:
            print(f"  fetched {i + 1} titles")
    pathlib.Path(out_path).write_text(json.dumps(result))
    return result
