"""Exercise 3 (GUIDE section 6): greedy page composition, Algorithm 2."""
import numpy as np

from wpo.baselines import slotting_pattern, slotting_patterns
from wpo.greedy import compose_pages, brute_force_page, regularized_value

CAT = np.array([0, 0, 0, 1, 1, 2, 2, 3])


def test_shapes_and_no_duplicates():
    R = np.random.default_rng(0).normal(size=(30, 8))
    pages = compose_pages(R, np.zeros((30, 4)), CAT, 5)
    assert pages.shape == (30, 5)
    assert all(len(set(p)) == 5 for p in pages)


def test_no_diversity_means_sort_by_relevance():
    R = np.random.default_rng(1).normal(size=(10, 8))
    pages = compose_pages(R, np.zeros((10, 4)), CAT, 4)
    assert np.array_equal(pages, np.argsort(-R, axis=1)[:, :4])


def test_negative_scores_still_fill_the_page():
    # Algorithm 2 initializes bestscore = 0; with all-negative scores that would place nothing.
    R = -np.ones((3, 8)) - np.arange(8)[None, :]
    pages = compose_pages(R, np.zeros((3, 4)), CAT, 3)
    assert np.array_equal(pages[0], [0, 1, 2])


def test_diversity_breaks_up_repeats():
    R = np.array([[1.0, 0.99, 0.98, 0.9, 0.1, 0.1, 0.1, 0.1]])
    D = np.array([[1.0, 1.0, 1.0, 1.0]])
    page = compose_pages(R, D, CAT, 2)[0]
    assert CAT[page[0]] != CAT[page[1]]


def test_large_dual_promotes_category():
    R = np.random.default_rng(2).uniform(size=(20, 8))
    xi = np.array([0.0, 0.0, 0.0, 10.0])
    pages = compose_pages(R, np.zeros((20, 4)), CAT, 3, xi=xi)
    assert np.all(CAT[pages[:, 0]] == 3)


def test_slotting_pins_are_respected():
    R = np.random.default_rng(3).uniform(size=(15, 8))
    R[:, CAT >= 2] -= 5.0          # pinned categories are unpopular: free slots won't use them up
    pattern = slotting_pattern(5, [0, 0, 1, 1])
    pages = compose_pages(R, np.zeros((15, 4)), CAT, 5, slot_category=pattern)
    for s, k in enumerate(pattern):
        if k >= 0:
            assert np.all(CAT[pages[:, s]] == k)
    per_user = slotting_patterns(15, 5, np.array([0, 0, 1.0, 0]), seed=0)
    pages = compose_pages(R, np.zeros((15, 4)), CAT, 5, slot_category=per_user)
    assert np.all((CAT[pages] == 2).sum(1) >= 1)


def test_greedy_is_near_optimal():
    rng = np.random.default_rng(4)
    for _ in range(20):
        r = rng.uniform(size=8)
        d = rng.uniform(0, 0.5, size=4)
        xi = rng.uniform(0, 0.2, size=4)
        g = compose_pages(r[None, :], d[None, :], CAT, 3, xi=xi)[0]
        _, best = brute_force_page(r, d, CAT, 3, range(8), xi)
        assert regularized_value(r, d, CAT, g, xi) >= (1 - 1 / np.e) * best - 1e-12


def test_exhausted_pin_falls_back_to_free_choice():
    R = np.zeros((1, 8)); R[0, 7] = 5.0      # the only category-3 widget is the favourite
    pages = compose_pages(R, np.zeros((1, 4)), CAT, 3, slot_category=np.array([-1, 3, -1]))
    assert pages[0, 0] == 7 and len(set(pages[0])) == 3
