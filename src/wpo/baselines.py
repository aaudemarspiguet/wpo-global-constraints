"""Baselines the paper compares against (Sections 4.1 and 5).

- CVR: pointwise relevance ranking, no diversity, no constraints (D = 0, xi = 0).
- Slotting: reserve slots for each constrained category on EVERY page.
- DIV (Teo et al. 2016 [26]): CVR scores + one global, non-personalized
  diversity weight; the paper hand-tunes per-category weights to hit the
  targets. We tune those per-category weights with the same dual solver,
  which is a fair, automated stand-in for "manual tuning".
"""
from __future__ import annotations

import numpy as np


def slotting_pattern(n_slots: int, pins_per_category, first_slot: int = 2, spacing: int = 3) -> np.ndarray:
    """Pin an integer number of slots per category, spread down the page.

    pins_per_category: (m,) ints. Returns (n_slots,) with a category index on
    pinned slots and -1 elsewhere.
    Example: n=8, pins=[0, 0, 1, 1] -> [-1, -1, 2, -1, -1, 3, -1, -1].
    """
    pins = np.asarray(pins_per_category, dtype=int)
    wanted = [k for k in range(len(pins)) for _ in range(pins[k])]
    if len(wanted) > n_slots:
        raise ValueError("targets need more pinned slots than the page has")
    order = list(range(first_slot, n_slots, spacing))
    order += [p for p in range(first_slot, n_slots) if p not in order]
    order += [p for p in range(first_slot) if p not in order]
    pattern = -np.ones(n_slots, dtype=int)
    for k, p in zip(wanted, order):
        pattern[p] = k
    return pattern


def slotting_patterns(n_users: int, n_slots: int, b, seed: int = 0) -> np.ndarray:
    """Per-page slotting that meets fractional targets b EXACTLY in expectation.

    Each page gets floor(b_k) pinned slots for category k, plus one more with
    probability frac(b_k). Returns (N, n_slots).
    """
    rng = np.random.default_rng(seed)
    b = np.asarray(b, dtype=float)
    base = np.floor(b).astype(int)
    extra = rng.random((n_users, len(b))) < (b - base)[None, :]
    pins = base[None, :] + extra
    return np.stack([slotting_pattern(n_slots, p) for p in pins])


def div_matrix(n_users: int, m: int, lam: float) -> np.ndarray:
    """DIV's diversity weight: the same lam for every user and category."""
    return np.full((n_users, m), lam)
