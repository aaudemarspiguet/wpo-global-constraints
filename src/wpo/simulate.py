"""Simulate pages being shown and clicked (infrastructure, not an exercise)."""
from __future__ import annotations

import numpy as np

from .world import World, Users, true_click_prob


def same_category_counts_above(widget_cat: np.ndarray, pages: np.ndarray, m: int) -> np.ndarray:
    """For every slot, how many rows of the SAME category sit above it. (N, n)."""
    N, n = pages.shape
    counts = np.zeros((N, m), dtype=int)
    out = np.zeros((N, n), dtype=int)
    rows = np.arange(N)
    for i in range(n):
        cat = widget_cat[pages[:, i]]
        out[:, i] = counts[rows, cat]
        counts[rows, cat] += 1
    return out


def true_page_probs(world: World, users: Users, pages: np.ndarray) -> np.ndarray:
    """TRUE stream probability of every slot on every page. (N, n).

    The sum over a row is the page's true expected number of streamed rows,
    the ground-truth "page reward" a real system can only estimate.
    """
    above = same_category_counts_above(world.widget_cat, pages, world.n_categories)
    return np.column_stack([
        true_click_prob(world, users, pages[:, i], above[:, i]) for i in range(pages.shape[1])
    ])


def random_pages(n_users: int, n_widgets: int, n_slots: int, seed: int = 0) -> np.ndarray:
    """A uniformly random logging policy: each user sees n random rows in random order.

    The paper trains on production logs (a biased logging policy). Uniform logging
    keeps our training data unbiased so you can focus on the method; see the
    guide's discussion of logging bias for what changes with real logs.
    """
    rng = np.random.default_rng(seed)
    return np.argsort(rng.random((n_users, n_widgets)), axis=1)[:, :n_slots]


def simulate_clicks(world: World, users: Users, pages: np.ndarray, seed: int = 0):
    """Sample binary outcomes y_i ~ Bernoulli(true p_i). Returns (p, y)."""
    rng = np.random.default_rng(seed)
    p = true_page_probs(world, users, pages)
    y = (rng.random(p.shape) < p).astype(float)
    return p, y
