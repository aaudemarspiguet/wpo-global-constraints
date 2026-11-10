"""Exercise 1 (GUIDE section 3): diversity features, Eq. (3)."""
import numpy as np
import pytest

from wpo.features import diversity_gain, diversity_features, page_feature_matrix, rho
from wpo.model import PageScorer
from wpo.greedy import model_page_values
from wpo.simulate import random_pages
from wpo.world import make_synthetic_world, sample_day


def test_gain_values():
    g = diversity_gain(np.array([0, 1, 2]))
    assert np.allclose(g, [np.log(2), np.log(1.5), np.log(4 / 3)])


def test_gain_diminishes():
    g = diversity_gain(np.arange(10))
    assert np.all(np.diff(g) < 0) and np.all(g > 0)


def test_gains_telescope_to_rho():
    # sum_i delta_rho = rho(I(C_n)): the identity behind the page-level model in Section 3
    for k in range(6):
        assert np.isclose(diversity_gain(np.arange(k)).sum(), rho(k))


def test_diversity_features_layout():
    ctx = np.array([[1.0, 2.0, 3.0], [1.0, 0.5, 0.0]])
    cats = np.array([2, 0])
    before = np.array([0, 1])
    F = diversity_features(ctx, cats, before, m=4).reshape(2, 3, 4)
    assert np.allclose(F[0, :, 2], ctx[0] * np.log(2))
    assert np.allclose(F[1, :, 0], ctx[1] * np.log(1.5))
    F[0, :, 2] = 0
    F[1, :, 0] = 0
    assert np.allclose(F, 0)


def test_features_agree_with_vectorized_scorer():
    w = make_synthetic_world(1, n_widgets=20, pool_size=500)
    u = sample_day(w, 0, 50)
    pages = random_pages(50, w.n_widgets, 5, seed=3)
    X = page_feature_matrix(w, u, pages, include_diversity=True)
    weights = np.random.default_rng(0).normal(size=X.shape[1])
    s = PageScorer(w, weights, True)
    via_features = (X @ weights).reshape(50, 5).sum(1)
    via_scorer = model_page_values(s.relevance_matrix(u), s.diversity_matrix(u), w.widget_cat, pages)
    assert np.allclose(via_features, via_scorer)
