"""Primal-dual optimization of the dual variables xi. Paper Section 4.2, Algorithm 1.

Alternate:
  (a) primal step: with xi fixed, compose every page independently, Eq. (9)
      (each page's problem decouples once xi is known);
  (b) dual step: projected gradient ASCENT on xi using the constraint miss,

      xi <- [ xi + eta_j * (b - avg I(C^u_n)) - eta_j * gamma_j * xi ]_+

  with diminishing step sizes eta_j = eta0 / j, gamma_j = gamma0 / sqrt(j).

The paper writes the gradient as (1/|U|) sum_u (b - sum_i I(c_i^u)), i.e.,
b minus the AVERAGE category counts per page. That is what we use.

Exercises in this file: dual_update, solve_duals.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .greedy import compose_pages
from .metrics import category_counts, miss_at_n, per_category_miss


def dual_update(xi: np.ndarray, b: np.ndarray, avg_counts: np.ndarray, eta: float, gamma: float) -> np.ndarray:
    """One projected (sub)gradient step on the dual, Algorithm 1 line 6."""
    raise NotImplementedError("TODO: Algorithm 1, line 6. GUIDE section 5.4")


def solve_duals(R: np.ndarray, D: np.ndarray, widget_cat: np.ndarray, n_slots: int, b: np.ndarray,
                eta0: float = 0.3, gamma0: float = 0.1, max_iter: int = 50, eps: float = 0.05,
                xi0: Optional[np.ndarray] = None,
                compose: Callable = compose_pages):
    """Algorithm 1. Returns (xi, pages, history).

    history is a list of dicts with keys: iter, miss, per_cat_miss, avg_counts, xi.
    Record BEFORE updating xi so history[0] is the unconstrained page (xi = xi0).
    Stop when miss_at_n <= eps (line 8) or after max_iter iterations.
    """
    raise NotImplementedError("TODO: Algorithm 1 loop. GUIDE section 5.5")
