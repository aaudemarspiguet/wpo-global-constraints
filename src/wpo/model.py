"""The page value model, paper Section 3 "Probability Model", Eq. (4).

The paper puts a Gaussian likelihood on the binary outcome y and a Gaussian
prior on each weight, then learns the posterior with online message passing
(Graepel et al. 2010 [15]). For a laptop-sized problem the posterior has a
closed form (Bishop, PRML Section 3.3):

    precision  A  = I / sigma0^2 + X^T X / beta0^2
    mean       mN = A^{-1} ( mu0 / sigma0^2 * 1 + X^T y / beta0^2 )
    covariance SN = A^{-1}

The posterior mean equals ridge regression with alpha = beta0^2 / sigma0^2
(plus the mu0 shift). Exercise in this file: BayesianLinearModel.fit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import relevance_dim
from .world import World, Users


class BayesianLinearModel:
    """y ~ N(x^T w, beta0^2), w_d ~ N(mu0, sigma0^2). Paper defaults: beta0=1, mu0=0.001, sigma0^2=1."""

    def __init__(self, beta0: float = 1.0, mu0: float = 0.001, sigma0: float = 1.0):
        self.beta0, self.mu0, self.sigma0 = beta0, mu0, sigma0
        self.mean = None
        self.cov = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianLinearModel":
        """Set self.mean (d,) and self.cov (d, d) to the posterior over w."""
        raise NotImplementedError("TODO: closed-form Gaussian posterior (Eq. 4). GUIDE section 3.5")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.mean

    def sample_weights(self, seed: int = 0) -> np.ndarray:
        """Draw w from the posterior. Algorithm 2, line 1 (Thompson sampling)."""
        rng = np.random.default_rng(seed)
        L = np.linalg.cholesky(self.cov + 1e-12 * np.eye(len(self.mean)))
        return self.mean + L @ rng.standard_normal(len(self.mean))


@dataclass
class PageScorer:
    """Turns a weight vector into the two matrices every composer needs.

    R[u, c]  relevance part phi(c, u)^T w_phi for every user and widget, (N, W)
    D[u, k]  diversity weight of category k for user u, = u^T Wdiv[:, k], (N, m)

    The score of placing widget c next on user u's page is then
        R[u, c] + D[u, cat(c)] * diversity_gain(count of cat(c) above)
    which is exactly Eq. (3) rearranged so it can be vectorized.
    """
    world: World
    weights: np.ndarray
    include_diversity: bool = True

    def _split(self):
        p = relevance_dim(self.world)
        return self.weights[:p], self.weights[p:]

    def relevance_matrix(self, users: Users) -> np.ndarray:
        w_phi, _ = self._split()
        wd = self.world
        cat = wd.widget_cat
        return (w_phi[0]
                + w_phi[1] * (users.taste_obs @ wd.widget_topic.T)
                + w_phi[2] * users.hist_prop[:, cat]
                + w_phi[3:][None, :])

    def diversity_matrix(self, users: Users) -> np.ndarray:
        m = self.world.n_categories
        if not self.include_diversity:
            return np.zeros((users.n, m))
        _, w_div = self._split()
        W_div = w_div.reshape(users.context.shape[1], m)
        return users.context @ W_div
