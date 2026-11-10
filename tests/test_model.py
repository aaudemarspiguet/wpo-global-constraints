"""Exercise 2 (GUIDE section 3.5): Bayesian linear regression, Eq. (4)."""
import numpy as np
from sklearn.linear_model import Ridge

from wpo.model import BayesianLinearModel


def test_posterior_mean_is_ridge():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0, 3.0]) + rng.normal(0, 0.5, 300)
    m = BayesianLinearModel(beta0=1.0, mu0=0.0, sigma0=0.5).fit(X, y)
    r = Ridge(alpha=1.0 ** 2 / 0.5 ** 2, fit_intercept=False).fit(X, y)
    assert np.allclose(m.mean, r.coef_, atol=1e-8)


def test_posterior_covariance_is_valid():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 4))
    m = BayesianLinearModel().fit(X, rng.normal(size=50))
    assert np.allclose(m.cov, m.cov.T)
    assert np.all(np.linalg.eigvalsh(m.cov) > 0)


def test_no_data_returns_prior():
    m = BayesianLinearModel(mu0=0.25).fit(np.zeros((10, 3)), np.ones(10))
    assert np.allclose(m.mean, 0.25)


def test_more_data_shrinks_uncertainty():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(1000, 3))
    y = rng.normal(size=1000)
    small = BayesianLinearModel().fit(X[:20], y[:20])
    big = BayesianLinearModel().fit(X, y)
    assert np.trace(big.cov) < np.trace(small.cov)
