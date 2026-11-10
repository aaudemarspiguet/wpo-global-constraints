"""Infrastructure that ships complete (simulator, baselines, data loaders)."""
import numpy as np
import pandas as pd

from wpo.baselines import slotting_patterns
from wpo.simulate import random_pages, true_page_probs, same_category_counts_above
from wpo.world import make_synthetic_world, sample_day, is_weekend


def test_world_and_day_sampling():
    w = make_synthetic_world(0, n_widgets=40, pool_size=1000)
    u1, u2 = sample_day(w, 2, 100), sample_day(w, 2, 100)
    assert np.array_equal(u1.pool_idx, u2.pool_idx)          # deterministic
    assert u1.context.shape == (100, 2 + w.n_categories)
    assert sample_day(w, 3, 10).weekend == is_weekend(3) is True


def test_true_probs_penalize_repeats():
    w = make_synthetic_world(0, n_widgets=40, pool_size=1000)
    u = sample_day(w, 0, 200)
    same = np.flatnonzero(w.widget_cat == 0)[:2]
    pages = np.tile(same, (200, 1))
    p = true_page_probs(w, u, pages)
    alone = true_page_probs(w, u, pages[:, 1:2])[:, 0]
    assert np.all(p[:, 1] < alone)


def test_counts_above():
    cat = np.array([0, 0, 1])
    out = same_category_counts_above(cat, np.array([[0, 2, 1]]), 2)
    assert np.array_equal(out, [[0, 0, 1]])


def test_fractional_slotting_hits_target_in_expectation():
    pats = slotting_patterns(20000, 8, np.array([0, 0, 1.5, 0.3]), seed=1)
    assert abs((pats == 2).sum(1).mean() - 1.5) < 0.02
    assert abs((pats == 3).sum(1).mean() - 0.3) < 0.02


def _tiny_movielens(d):
    rng = np.random.default_rng(0)
    genres = ["Action", "Comedy", "Drama", "Horror"]
    movies = pd.DataFrame({
        "movieId": np.arange(1, 121),
        "title": [f"Movie {i}" for i in range(1, 121)],
        "genres": ["|".join(rng.choice(genres, size=rng.integers(1, 3), replace=False)) for _ in range(120)],
    })
    links = pd.DataFrame({"movieId": movies.movieId, "imdbId": movies.movieId + 1000,
                          "tmdbId": movies.movieId + 5000})
    rows = []
    for uid in range(1, 41):
        for mid in rng.choice(120, size=30, replace=False) + 1:
            rows.append((uid, mid, float(rng.integers(1, 11)) / 2, 1_600_000_000 + int(rng.integers(0, 10**7))))
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    d.mkdir()
    movies.to_csv(d / "movies.csv", index=False)
    links.to_csv(d / "links.csv", index=False)
    ratings.to_csv(d / "ratings.csv", index=False)


def test_movielens_world(tmp_path):
    import json
    from wpo.data.movielens import world_from_movielens
    d = tmp_path / "ml"
    _tiny_movielens(d)
    w = world_from_movielens(d, min_titles=3, min_user_ratings=10)
    assert w.n_categories == 3 and w.n_widgets > 5 and w.pseudo_categories
    assert np.allclose(w.pool_taste.sum(1), 1)
    prov = {str(5000 + i): ({"flatrate": ["Amazon Prime Video"], "rent": [], "buy": []} if i % 3 == 0 else
                            {"flatrate": [], "rent": ["Apple TV"], "buy": []} if i % 3 == 1 else
                            {"flatrate": ["Other"], "rent": [], "buy": []}) for i in range(1, 121)}
    (tmp_path / "p.json").write_text(json.dumps(prov))
    w2 = world_from_movielens(d, providers_json=tmp_path / "p.json", min_titles=3, min_user_ratings=10)
    assert not w2.pseudo_categories
    assert set(np.unique(w2.widget_cat)) == {0, 1, 2}
    u = sample_day(w2, 0, 30)
    pages = random_pages(30, w2.n_widgets, 4, seed=0)
    assert true_page_probs(w2, u, pages).shape == (30, 4)
