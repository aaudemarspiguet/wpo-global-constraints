"""ContentWise Impressions (Perez Maurera et al., CIKM 2020): REAL carousel impressions.

This is the closest public analogue to the paper's homepage logs: rows of
recommendations shown to users of a real streaming (OTT) service, with the
row's position on screen. Use it to CALIBRATE the simulator (category mix by
row position, which categories get engaged with) and to sanity-check a
row-level relevance model. It does not let you run counterfactual whole-page
experiments, because you cannot observe what users would have done with a
page that was never shown. That is why the core experiments use the simulator.

SCHEMA WARNING. The column names below are what the dataset's documentation
describes as of this writing. Verify them against the README shipped inside
the download, and edit the constants if anything differs. check_schema()
prints what is actually there.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

TABLES = {
    "interactions": "interactions",
    "impressions_direct": "impressions-direct-link",
    "impressions_non_direct": "impressions-non-direct-link",
}

# Expected columns (verify!). Edit here if your copy differs.
COL_USER = "user_id"
COL_ITEM = "item_id"
COL_SERIES = "series_id"
COL_ITEM_TYPE = "item_type"
COL_INTERACTION_TYPE = "interaction_type"
COL_RECOMMENDATION_ID = "recommendation_id"
COL_ROW_POSITION = "row_position"
COL_LIST_LENGTH = "recommendation_list_length"
COL_RECOMMENDED = "recommended_series_list"

EXPECTED = {
    "interactions": [COL_USER, COL_ITEM, COL_SERIES, COL_ITEM_TYPE, COL_INTERACTION_TYPE, COL_RECOMMENDATION_ID],
    "impressions_direct": [COL_RECOMMENDATION_ID, COL_ROW_POSITION, COL_LIST_LENGTH, COL_RECOMMENDED],
    "impressions_non_direct": [COL_USER, COL_ROW_POSITION, COL_LIST_LENGTH, COL_RECOMMENDED],
}


def _read_any(path: pathlib.Path) -> pd.DataFrame:
    if path.is_dir():
        parts = sorted(list(path.rglob("*.parquet")) + list(path.rglob("*.csv")))
        if not parts:
            raise FileNotFoundError(f"no parquet/csv files under {path}")
        return pd.concat([_read_any(p) for p in parts], ignore_index=True)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def find_table(data_dir, stem: str) -> pathlib.Path:
    d = pathlib.Path(data_dir)
    hits = [p for p in d.rglob(f"{stem}*") if p.suffix in (".parquet", ".csv") or p.is_dir()]
    if not hits:
        raise FileNotFoundError(f"could not find '{stem}' under {d}")
    return sorted(hits, key=lambda p: len(str(p)))[0]


def load_tables(data_dir, nrows_sample: int | None = None, seed: int = 0) -> dict:
    """Load the three tables. nrows_sample subsamples rows (the full data is large)."""
    out = {}
    for key, stem in TABLES.items():
        df = _read_any(find_table(data_dir, stem))
        if df.index.name and df.index.name not in df.columns:
            df = df.reset_index()
        if nrows_sample and len(df) > nrows_sample:
            df = df.sample(nrows_sample, random_state=seed)
        out[key] = df
    return out


def check_schema(tables: dict) -> bool:
    ok = True
    for key, cols in EXPECTED.items():
        have = list(tables[key].columns)
        missing = [c for c in cols if c not in have]
        print(f"{key}: {len(tables[key]):,} rows; columns = {have}")
        if missing:
            ok = False
            print(f"  MISSING expected columns {missing}; edit the constants in wpo/data/contentwise.py")
    return ok


def series_type_map(interactions: pd.DataFrame) -> pd.Series:
    """series_id -> its most common item_type (our stand-in for the paper's 'category')."""
    return (interactions.groupby(COL_SERIES)[COL_ITEM_TYPE]
            .agg(lambda s: s.value_counts().index[0]))


def _as_list(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    if isinstance(x, str):
        return [int(t) for t in x.strip("[]").replace(",", " ").split() if t.strip()]
    return []


def row_categories(impressions: pd.DataFrame, s2type: pd.Series) -> pd.Series:
    """Each impressed row -> majority item_type of the series it recommended (a 'widget category')."""
    def majority(lst):
        types = s2type.reindex(_as_list(lst)).dropna()
        return types.value_counts().index[0] if len(types) else np.nan
    return impressions[COL_RECOMMENDED].map(majority)


def calibration_report(tables: dict) -> pd.DataFrame:
    """Category mix of impressed rows by row position, for linked vs unlinked impressions.

    Paper Section 5 sets targets b to replicate the production balance; this
    table is the real-data version of that balance. A row in the
    impressions-direct-link table had at least one interaction linked to it,
    so the linked share per category is a rough engagement signal (with
    selection bias; discuss it in your write-up).
    """
    s2t = series_type_map(tables["interactions"])
    frames = []
    for key, linked in [("impressions_direct", True), ("impressions_non_direct", False)]:
        imp = tables[key].copy()
        imp["row_category"] = row_categories(imp, s2t)
        imp["linked"] = linked
        frames.append(imp[[COL_ROW_POSITION, "row_category", "linked"]])
    allimp = pd.concat(frames, ignore_index=True).dropna(subset=["row_category"])
    mix = pd.crosstab(allimp[COL_ROW_POSITION], allimp["row_category"], normalize="index")
    linked_rate = allimp.groupby("row_category")["linked"].mean().rename("linked_share")
    print("Linked share by category:\n", linked_rate.round(4))
    return mix
