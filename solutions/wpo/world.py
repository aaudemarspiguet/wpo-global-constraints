"""The simulated "world": widgets, users, and the TRUE click model.

This module is infrastructure, not an exercise. It plays the role of the real
Amazon Video customers in Ding et al. (KDD '19), Section 5. Because we own the
ground truth, we can measure things the paper could not (e.g., the *true*
expected clicks of any page a policy composes).

Paper notation used throughout the repo
---------------------------------------
u          page context (customer, device, time)         -> Users.context[i]
c          widget context (id, items, category)          -> widget index
C^u_n      the ordered top-n widgets on a page           -> pages[i, :n]
I(c)       one-hot category indicator, I(c) in {0,1}^m   -> widget_cat[c]
U^t        all page contexts in time period t            -> Users for day t
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

CATEGORY_NAMES = ["prime_movie", "prime_tv", "rent_buy", "channel"]
CATEGORY_LABELS = {
    "prime_movie": "movies included with Prime",
    "prime_tv": "series included with Prime",
    "rent_buy": "to rent or buy",
    "channel": "on add-on channels",
}
GENRES = ["Action", "Comedy", "Drama", "Documentary", "Kids", "Sci-fi"]

# Days whose (day % 7) is in this set behave like weekends. With evaluation days
# 7..13 this makes evaluation days 4 and 5 (1-indexed) weekends, as in Fig. 3.
WEEKEND_MOD7 = {3, 4}


def is_weekend(day: int) -> bool:
    return (day % 7) in WEEKEND_MOD7


@dataclass
class World:
    categories: List[str]
    genres: List[str]
    widget_cat: np.ndarray        # (W,) int category of each widget
    widget_topic: np.ndarray      # (W, G) genre mix of the widget's items
    widget_quality: np.ndarray    # (W,) intrinsic appeal
    widget_names: List[str]
    pool_taste: np.ndarray        # (P, G) true genre taste of each potential user
    pool_aff: np.ndarray          # (P, m) true category affinity (logit units)
    pool_weekend_weight: np.ndarray  # (P,) relative sampling weight on weekends
    click_bias: float = -2.2
    taste_scale: float = 4.0
    repeat_penalty: float = 0.6
    obs_noise: float = 0.35

    @property
    def n_widgets(self) -> int:
        return len(self.widget_cat)

    @property
    def n_categories(self) -> int:
        return len(self.categories)

    @property
    def n_genres(self) -> int:
        return self.widget_topic.shape[1]


@dataclass
class Users:
    """All page contexts u in one time period U^t (here: one day)."""
    day: int
    weekend: bool
    pool_idx: np.ndarray   # (N,) index into the world's user pool
    taste: np.ndarray      # (N, G) TRUE taste (hidden from models)
    aff: np.ndarray        # (N, m) TRUE category affinity (hidden)
    taste_obs: np.ndarray  # (N, G) noisy observed taste (models may use)
    hist_prop: np.ndarray  # (N, m) observed historical propensity in [0, 1]
    context: np.ndarray    # (N, d_u) the context vector u = [1, hist_prop, weekend]

    @property
    def n(self) -> int:
        return len(self.pool_idx)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _widget_names(cats, topic, categories, genres):
    names, seen = [], {}
    for c, t in zip(cats, topic):
        g = genres[int(np.argmax(t))]
        base = f"{g} {CATEGORY_LABELS.get(categories[c], categories[c])}"
        seen[base] = seen.get(base, 0) + 1
        names.append(base if seen[base] == 1 else f"{base} ({seen[base]})")
    return names


def make_synthetic_world(seed: int = 0, n_widgets: int = 60, pool_size: int = 20000) -> World:
    """Build a fully synthetic world shaped like the paper's Prime Video setting.

    Categories mirror Section 1: subscription content (SVOD movies / TV),
    transactional rent-or-buy (TVOD) and third-party channels (3P).
    Many users never rent or subscribe to channels; a minority love them.
    That heterogeneity is exactly what makes *global* constraints beat slotting.
    """
    rng = np.random.default_rng(seed)
    m, G = len(CATEGORY_NAMES), len(GENRES)

    shares = np.array([0.33, 0.25, 0.22, 0.20])
    counts = np.floor(shares * n_widgets).astype(int)
    counts[0] += n_widgets - counts.sum()
    widget_cat = np.repeat(np.arange(m), counts)
    widget_topic = rng.dirichlet(np.full(G, 0.3), size=n_widgets)
    widget_quality = rng.normal(0.0, 0.3, size=n_widgets)

    pool_taste = rng.dirichlet(np.full(G, 0.5), size=pool_size)
    aff = np.zeros((pool_size, m))
    aff[:, 0] = rng.normal(0.3, 0.4, pool_size)
    aff[:, 1] = rng.normal(0.0, 0.5, pool_size)

    def segmented(means, probs):
        seg = rng.choice(len(means), size=pool_size, p=probs)
        return np.asarray(means)[seg] + rng.normal(0, 0.3, pool_size)

    aff[:, 2] = segmented([-2.0, -0.3, 0.8], [0.45, 0.40, 0.15])   # rent / buy
    aff[:, 3] = segmented([-1.8, -0.2, 0.9], [0.55, 0.30, 0.15])   # channels

    weekend_w = 1.0 + 1.5 * (aff[:, 3] > 0) + 0.5 * (aff[:, 1] > 0.3)

    return World(
        categories=list(CATEGORY_NAMES), genres=list(GENRES),
        widget_cat=widget_cat, widget_topic=widget_topic, widget_quality=widget_quality,
        widget_names=_widget_names(widget_cat, widget_topic, CATEGORY_NAMES, GENRES),
        pool_taste=pool_taste, pool_aff=aff, pool_weekend_weight=weekend_w,
    )


def sample_day(world: World, day: int, n_users: int, seed: int = 0) -> Users:
    """Sample the page contexts U^t for one day. Weekends shift the user mix."""
    rng = np.random.default_rng([seed, day])
    P = world.pool_taste.shape[0]
    w = world.pool_weekend_weight if is_weekend(day) else np.ones(P)
    idx = rng.choice(P, size=n_users, replace=True, p=w / w.sum())
    taste = world.pool_taste[idx]
    aff = world.pool_aff[idx]
    taste_obs = np.clip(taste + rng.normal(0, world.obs_noise * taste.std(), taste.shape), 0, None)
    taste_obs = taste_obs / np.maximum(taste_obs.sum(1, keepdims=True), 1e-9)
    hist_prop = _sigmoid(aff + rng.normal(0, world.obs_noise, aff.shape))
    wk = float(is_weekend(day))
    context = np.hstack([np.ones((n_users, 1)), hist_prop, np.full((n_users, 1), wk)])
    return Users(day, bool(wk), idx, taste, aff, taste_obs, hist_prop, context)


def true_click_prob(world: World, users: Users, widgets: np.ndarray, same_cat_above: np.ndarray) -> np.ndarray:
    """TRUE P(stream from this row | rows above). Hidden from all models.

    widgets: (N,) widget index shown to each user at the current slot.
    same_cat_above: (N,) how many rows of the same category are already above.
    Users who love a category tolerate repeats of it (smaller penalty), which is
    why the paper interacts diversity features with context (u x delta-rho).
    """
    N = users.n
    cat = world.widget_cat[widgets]
    rel = (users.taste * world.widget_topic[widgets]).sum(1)
    a = users.aff[np.arange(N), cat]
    tolerance = np.exp(-0.5 * np.maximum(a, 0.0))
    logit = (world.click_bias + world.taste_scale * rel + a
             + world.widget_quality[widgets]
             - world.repeat_penalty * tolerance * same_cat_above)
    return _sigmoid(logit)
