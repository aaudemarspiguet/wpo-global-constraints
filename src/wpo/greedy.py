"""Greedy page composition with dual variables. Paper Section 4.3, Algorithm 2.

For each slot i = 1..n, pick the remaining widget a maximizing

    s_a = f(a | C_{i-1}, u, w) + xi^T I(a)

i.e., the model's score given the rows above, plus the dual "bonus" of a's
category. Because f is monotone submodular (when the diversity weights are
non-negative) and xi^T I(a) is modular with xi >= 0, greedy is within
(1 - 1/e) of the optimum of Eq. (9) (Nemhauser, Wolsey & Fisher 1978).

Exercise in this file: compose_pages.
"""
from __future__ import annotations

import itertools
from typing import Optional

import numpy as np

from .features import diversity_gain


def compose_pages(R: np.ndarray, D: np.ndarray, widget_cat: np.ndarray, n_slots: int,
                  xi: Optional[np.ndarray] = None, slot_category: Optional[np.ndarray] = None) -> np.ndarray:
    """Greedily build one page per user, vectorized over users. Returns (N, n_slots).

    R: (N, W) relevance scores. D: (N, m) per-user diversity weights.
    xi: (m,) dual variables, or None for the unconstrained page (xi = 0).
    slot_category: optional (n_slots,) or (N, n_slots) int array; entry k >= 0
        pins that slot to category k (the slotting baseline), -1 leaves it free.

    Implementation notes:
      * Never place the same widget twice on a page.
      * Initialize the best score to -inf, NOT 0 as Algorithm 2 line 4 does:
        with 0, a page whose candidates all score negative would get no widget.
      * If a pinned slot's category has no unused widget left, fill it freely.
    """
    raise NotImplementedError("TODO: Algorithm 2, vectorized across users. GUIDE section 6.2")


def model_page_values(R: np.ndarray, D: np.ndarray, widget_cat: np.ndarray, pages: np.ndarray) -> np.ndarray:
    """Model-estimated value f(C_n | u, w) of each page, Eq. (2). Returns (N,)."""
    N, n = pages.shape
    m = D.shape[1]
    rows = np.arange(N)
    counts = np.zeros((N, m), dtype=int)
    total = np.zeros(N)
    for i in range(n):
        a = pages[:, i]
        k = widget_cat[a]
        total += R[rows, a] + D[rows, k] * diversity_gain(counts[rows, k])
        counts[rows, k] += 1
    return total


def regularized_value(r: np.ndarray, d: np.ndarray, widget_cat: np.ndarray, page, xi) -> float:
    """Eq. (9) objective for ONE user's page: f(page) + xi^T I(page)."""
    counts = np.zeros(len(d), dtype=int)
    v = 0.0
    for a in page:
        k = widget_cat[a]
        v += r[a] + d[k] * float(diversity_gain(counts[k])) + (0.0 if xi is None else xi[k])
        counts[k] += 1
    return v


def brute_force_page(r: np.ndarray, d: np.ndarray, widget_cat: np.ndarray, n_slots: int,
                     candidates, xi=None):
    """Exhaustively search ordered pages over a SMALL candidate set. For checking greedy."""
    best, best_v = None, -np.inf
    for page in itertools.permutations(candidates, n_slots):
        v = regularized_value(r, d, widget_cat, page, xi)
        if v > best_v:
            best, best_v = page, v
    return list(best), best_v
