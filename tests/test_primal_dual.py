"""Exercise 4 (GUIDE section 5): primal-dual updates, Algorithm 1."""
import numpy as np

from wpo.greedy import compose_pages
from wpo.metrics import category_counts, miss_at_n
from wpo.primal_dual import dual_update, solve_duals


def test_update_direction_and_projection():
    b = np.array([0.0, 1.0])
    xi = np.array([0.0, 0.5])
    under = dual_update(xi, b, np.array([2.0, 0.5]), eta=0.1, gamma=0.0)
    assert np.isclose(under[1], 0.55)          # below target -> bonus grows
    over = dual_update(xi, b, np.array([2.0, 3.0]), eta=0.1, gamma=0.0)
    assert np.isclose(over[1], 0.3)            # above target -> bonus shrinks
    assert under[0] == 0.0 and over[0] == 0.0  # projected onto xi >= 0


def test_regularization_shrinks():
    xi = np.array([1.0])
    out = dual_update(xi, np.array([1.0]), np.array([1.0]), eta=0.1, gamma=0.5)
    assert np.isclose(out[0], 0.95)


def _toy(N=400, seed=0):
    rng = np.random.default_rng(seed)
    cat = np.repeat(np.arange(3), 5)
    R = rng.uniform(size=(N, 15))
    R[:, cat == 2] -= 0.4                  # category 2 is unpopular
    lovers = rng.random(N) < 0.3
    R[lovers][:, cat == 2] += 0.6
    R[np.ix_(lovers, cat == 2)] += 0.6
    return R, np.zeros((N, 3)), cat


def test_solver_reduces_miss_and_records_history():
    R, D, cat = _toy()
    b = np.array([0.0, 0.0, 1.0])
    free = compose_pages(R, D, cat, 4)
    start = miss_at_n(category_counts(free, cat, 3).mean(0), b)
    xi, pages, hist = solve_duals(R, D, cat, 4, b, eta0=0.5, gamma0=0.0, eps=0.01, max_iter=100)
    assert np.allclose(hist[0]["xi"], 0) and np.isclose(hist[0]["miss"], start)
    assert pages.shape == (400, 4)
    assert hist[-1]["miss"] <= 0.01 < start
    assert xi[2] > 0 and xi[0] == 0
