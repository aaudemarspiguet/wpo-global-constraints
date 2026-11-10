"""Exercise 5 (GUIDE section 7.2): evaluation metrics."""
import numpy as np

from wpo.metrics import div_pair, miss_at_n, coverage_by_bucket

CAT = np.array([0, 1, 2, 3])


def test_div_pair_extremes_and_middle():
    same = div_pair(np.array([[0, 0, 0]]), np.array([5, 5, 5, 5]) * 0, 4)
    assert np.isclose(same, 0.0)
    assert np.isclose(div_pair(np.array([[0, 1, 2, 3]]), CAT, 4), 1.0)
    cat = np.array([0, 0, 1, 1])
    assert np.isclose(div_pair(np.array([[0, 1, 2, 3]]), cat, 2), 1 - 2 / 6)


def test_miss_at_n():
    b = np.array([0.0, 0.0, 1.8, 1.8])
    assert np.isclose(miss_at_n(np.array([3, 3, 0.9, 2.0]), b), 0.25)
    assert miss_at_n(np.array([1.0, 1.0, 1.0, 1.0]), np.zeros(4)) == 0.0


def test_coverage_by_bucket():
    out = coverage_by_bucket(np.array([0, 0, 2, 4]), np.array([0, 0, 1, 1]), ["a", "b"], 2.0)
    assert np.isclose(out["a"], -100.0) and np.isclose(out["b"], 50.0)
