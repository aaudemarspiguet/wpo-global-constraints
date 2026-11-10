"""Features for the page scoring model. Paper Section 3, Eq. (2)-(3).

f(c_i | C_{i-1}, u, w) = [ phi(c_i, u)^T , (u x delta_rho(c_i, C_{i-1}))^T ] w

- phi(c, u): contextual relevance features (the paper assumes they are given).
- delta_rho: the diminishing-returns diversity gain of adding c_i given the
  category counts of the rows above it, with rho(x) = log(1 + x).
- u x delta_rho: the outer product with the context, so diversity can matter
  more for some users than others.

Exercises in this file: diversity_gain, diversity_features.
"""
from __future__ import annotations

import numpy as np

from .world import World, Users
from .simulate import same_category_counts_above


def rho(x):
    """The one-dimensional submodular function the paper uses: rho(x) = log(1 + x)."""
    return np.log1p(x)


def diversity_gain(count_before):
    """delta_rho for ONE category: rho(count + 1) - rho(count).

    count_before: array of how many rows of this widget's category are already
    above. Returns an array of the same shape. Examples: 0 -> log 2 ~ 0.693,
    1 -> log(3/2) ~ 0.405, 2 -> log(4/3) ~ 0.288.
    """
    # >>> SOLUTION: Eq. (3) diversity gain, rho(n+1) - rho(n). GUIDE section 3.3
    c = np.asarray(count_before, dtype=float)
    return rho(c + 1.0) - rho(c)
    # <<< SOLUTION


def relevance_dim(world: World) -> int:
    return 3 + world.n_widgets


def relevance_features(world: World, users: Users, widgets: np.ndarray) -> np.ndarray:
    """phi(c, u) for one widget per user. Returns (N, 3 + W).

    Layout: [bias, taste_obs . topic, hist_prop[category], one-hot(widget id)].
    These are deliberately simple; the paper treats phi as given.
    """
    N = users.n
    cat = world.widget_cat[widgets]
    out = np.zeros((N, relevance_dim(world)))
    out[:, 0] = 1.0
    out[:, 1] = (users.taste_obs * world.widget_topic[widgets]).sum(1)
    out[:, 2] = users.hist_prop[np.arange(N), cat]
    out[np.arange(N), 3 + widgets] = 1.0
    return out


def diversity_features(context: np.ndarray, widget_cats: np.ndarray, count_before: np.ndarray, m: int) -> np.ndarray:
    """The context-interacted diversity features u x delta_rho. Returns (N, d_u * m).

    delta_rho is an m-dim vector that is zero except at the widget's own
    category, where it equals diversity_gain(count_before). Take its outer
    product with the context u (shape d_u) and flatten row-major, so feature
    index j * m + k means "context feature j times gain in category k".
    """
    # >>> SOLUTION: outer product u (x) delta_rho, flattened. GUIDE section 3.4
    N, d_u = context.shape
    gain = diversity_gain(count_before)
    out = np.zeros((N, d_u, m))
    out[np.arange(N), :, widget_cats] = context * gain[:, None]
    return out.reshape(N, d_u * m)
    # <<< SOLUTION


def page_feature_matrix(world: World, users: Users, pages: np.ndarray, include_diversity: bool = True) -> np.ndarray:
    """Stack features for every (user, slot) of logged pages. Returns (N * n, d).

    Rows are ordered user-major: row i * n + s is user i, slot s, matching
    y.reshape(-1) for a (N, n) outcome matrix.
    """
    N, n = pages.shape
    m = world.n_categories
    above = same_category_counts_above(world.widget_cat, pages, m)
    blocks = []
    for s in range(n):
        w = pages[:, s]
        parts = [relevance_features(world, users, w)]
        if include_diversity:
            parts.append(diversity_features(users.context, world.widget_cat[w], above[:, s], m))
        blocks.append(np.hstack(parts))
    X = np.stack(blocks, axis=1)          # (N, n, d)
    return X.reshape(N * n, -1)
