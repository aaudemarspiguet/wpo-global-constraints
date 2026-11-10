"""Evaluation metrics from paper Section 5 (plus ground-truth versions the paper can't compute).

Exercises in this file: div_pair, miss_at_n.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auc(y: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y, scores))


def category_counts(pages: np.ndarray, widget_cat: np.ndarray, m: int) -> np.ndarray:
    """I(C_n^u) for every page: (N, m) counts of each category on the page."""
    cats = widget_cat[pages]
    return np.stack([(cats == k).sum(1) for k in range(m)], axis=1)


def expected_precision_at_k(p_true: np.ndarray, k: int) -> float:
    """Ground-truth Precision@K: mean TRUE stream probability of the top-K rows.

    The paper counts observed positives in the top-K of logged pages; in a
    simulator we can use the expectation directly, which has no noise.
    """
    return float(p_true[:, :k].mean())


def div_pair(pages: np.ndarray, widget_cat: np.ndarray, m: int) -> float:
    """Div-pair@n: average fraction of row PAIRS on a page that are in DIFFERENT categories.

    For a page with category counts I_1..I_m, the number of same-category pairs
    is sum_k I_k (I_k - 1) / 2 out of n (n - 1) / 2 pairs. Div-pair = 1 - ratio,
    averaged over pages. (The printed formula in the paper is ambiguous; this
    is the standard pairwise-dissimilarity reading. See GUIDE section 7.2.)
    """
    raise NotImplementedError("TODO: pairwise dissimilarity. GUIDE section 7.2")


def miss_at_n(avg_counts: np.ndarray, b: np.ndarray) -> float:
    """Eq. (10): mean relative shortfall from the impression targets.

    avg_counts: (m,) average rows per page of each category across U^t.
    b: (m,) minimum targets. Categories with b_s = 0 are unconstrained and are
    skipped (the paper's 1 - x / 0 is undefined for them).
    Returns mean over constrained s of max(0, 1 - avg_s / b_s).
    """
    raise NotImplementedError("TODO: Eq. (10). GUIDE section 7.2")


def per_category_miss(avg_counts: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(b, dtype=float)
    mask = b > 0
    out[mask] = np.maximum(0.0, 1.0 - avg_counts[mask] / b[mask])
    return out


def coverage_by_bucket(cat_counts: np.ndarray, bucket: np.ndarray, labels, b_k: float):
    """Table 3: % difference between a bucket's average impressions and the target b_k."""
    out = {}
    for j, lab in enumerate(labels):
        sel = bucket == j
        out[lab] = float((cat_counts[sel].mean() / b_k - 1.0) * 100.0) if sel.any() else float("nan")
    return out


def pct_lift(new: float, base: float) -> float:
    return float((new - base) / abs(base) * 100.0) if base != 0 else float("nan")
