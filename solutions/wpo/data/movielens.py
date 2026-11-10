"""Build a semi-synthetic World from MovieLens (+ optional TMDB watch providers).

REAL parts:   the catalog (titles, genres), user tastes (from ratings), when
              users are active (weekday vs weekend timestamps), and, if you
              fetched them, real stream / rent / buy availability from TMDB.
SIMULATED:    page impressions and clicks (MovieLens has no homepage logs).

Files expected in ml_dir (ml-latest-small or ml-1m converted to CSV):
    movies.csv   movieId,title,genres          (genres like "Action|Comedy")
    ratings.csv  userId,movieId,rating,timestamp
    links.csv    movieId,imdbId,tmdbId

Category mapping (MovieLens is movies only, so m = 3):
    prime_movie  available by subscription on the "house" service
    rent_buy     available to rent or buy (TVOD)
    channel      available by subscription on some OTHER service (3P-like)
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Optional

import numpy as np
import pandas as pd

from ..world import World, CATEGORY_LABELS

ML_CATEGORIES = ["prime_movie", "rent_buy", "channel"]


def _pseudo_category(movie_id: int) -> str:
    """Deterministic stand-in when no provider data is available. NOT real availability."""
    h = int(hashlib.md5(str(movie_id).encode()).hexdigest(), 16) % 100
    return "prime_movie" if h < 50 else ("rent_buy" if h < 80 else "channel")


def _real_category(prov: Optional[dict], house: str) -> Optional[str]:
    if not prov:
        return None
    flat = set(prov.get("flatrate", []))
    if house in flat:
        return "prime_movie"
    if prov.get("rent") or prov.get("buy"):
        return "rent_buy"
    if flat:
        return "channel"
    return None


def load_movielens(ml_dir) -> tuple:
    d = pathlib.Path(ml_dir)
    movies = pd.read_csv(d / "movies.csv")
    ratings = pd.read_csv(d / "ratings.csv")
    links = pd.read_csv(d / "links.csv")
    return movies, ratings, links


def world_from_movielens(ml_dir, providers_json=None, house_provider: str = "Amazon Prime Video",
                         n_genres: int = 8, min_titles: int = 6, items_per_widget: int = 20,
                         min_user_ratings: int = 20, seed: int = 0) -> World:
    movies, ratings, links = load_movielens(ml_dir)
    movies = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")

    # --- categories -------------------------------------------------------
    if providers_json is not None:
        prov = json.loads(pathlib.Path(providers_json).read_text())
        movies["category"] = [
            _real_category(prov.get(str(int(t))) if pd.notna(t) else None, house_provider)
            for t in movies["tmdbId"]]
        pseudo = False
    else:
        movies["category"] = [_pseudo_category(int(i)) for i in movies["movieId"]]
        pseudo = True
    movies = movies.dropna(subset=["category"]).copy()

    # --- genres -----------------------------------------------------------
    g_lists = movies["genres"].fillna("").str.split("|")
    counts = pd.Series([g for gs in g_lists for g in gs if g and g != "(no genres listed)"]).value_counts()
    genres = list(counts.index[:n_genres])
    gidx = {g: i for i, g in enumerate(genres)}
    multi_hot = np.zeros((len(movies), len(genres)))
    for r, gs in enumerate(g_lists):
        for g in gs:
            if g in gidx:
                multi_hot[r, gidx[g]] = 1.0
    movies = movies.reset_index(drop=True)
    pop = ratings.groupby("movieId").size()
    movies["pop"] = movies["movieId"].map(pop).fillna(0).values

    # --- widgets: one row per (category, genre) with enough titles, + "Popular" per category
    cat_index = {c: k for k, c in enumerate(ML_CATEGORIES)}
    w_cat, w_topic, w_qual, w_names = [], [], [], []

    def add_widget(mask, k, name):
        sel = movies[mask].sort_values("pop", ascending=False).head(items_per_widget)
        if len(sel) < min_titles:
            return
        topic = multi_hot[sel.index].sum(0) + 1e-3
        w_cat.append(k)
        w_topic.append(topic / topic.sum())
        w_qual.append(np.log1p(sel["pop"].mean()))
        w_names.append(name)

    for c, k in cat_index.items():
        in_cat = movies["category"] == c
        add_widget(in_cat, k, f"Popular {CATEGORY_LABELS[c]}")
        for g in genres:
            add_widget(in_cat & (multi_hot[:, gidx[g]] > 0), k, f"{g} {CATEGORY_LABELS[c]}")
    if not w_cat:
        raise ValueError("no widgets built; lower min_titles or check the category mapping")
    w_qual = np.asarray(w_qual)
    w_qual = 0.3 * (w_qual - w_qual.mean()) / (w_qual.std() + 1e-9)

    # --- users: taste from ratings, category affinity from what they rate ---
    r = ratings.merge(movies[["movieId", "category"]], on="movieId", how="inner")
    row_of = pd.Series(np.arange(len(movies)), index=movies["movieId"])
    r = r[r.groupby("userId")["movieId"].transform("size") >= min_user_ratings]
    if r.empty:
        raise ValueError("no users with enough ratings; lower min_user_ratings")
    users = np.sort(r["userId"].unique())
    uidx = pd.Series(np.arange(len(users)), index=users)
    weight = np.clip(r["rating"] - r.groupby("userId")["rating"].transform("mean") + 1.0, 0, None)
    taste = np.zeros((len(users), len(genres)))
    np.add.at(taste, uidx[r["userId"]].values, multi_hot[row_of[r["movieId"]].values] * weight.values[:, None])
    taste += 1e-3
    taste /= taste.sum(1, keepdims=True)

    # ASSUMPTION (state it in your write-up): a user who rates more titles from a
    # category than the population does has a higher affinity for that category.
    share = pd.crosstab(r["userId"], r["category"]).reindex(columns=ML_CATEGORIES, fill_value=0)
    share = share.div(share.sum(axis=1), axis=0).reindex(users).values
    base = r["category"].value_counts(normalize=True).reindex(ML_CATEGORIES).fillna(0).values
    aff = 1.5 * np.log((share + 0.02) / (base + 0.02))
    aff[:, cat_index["rent_buy"]] -= 0.5      # buying costs money: lower baseline appetite
    aff[:, cat_index["channel"]] -= 0.7       # so do extra subscriptions

    ts = pd.to_datetime(r["timestamp"], unit="s")
    weekend_share = (ts.dt.dayofweek >= 5).groupby(r["userId"]).mean().reindex(users).values
    weekend_w = 1.0 + 2.0 * weekend_share

    world = World(categories=list(ML_CATEGORIES), genres=genres,
                  widget_cat=np.asarray(w_cat), widget_topic=np.asarray(w_topic),
                  widget_quality=w_qual, widget_names=w_names,
                  pool_taste=taste, pool_aff=aff, pool_weekend_weight=weekend_w)
    world.pseudo_categories = pseudo  # type: ignore[attr-defined]
    return world
