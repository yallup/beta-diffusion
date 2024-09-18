import distrax
import jax
import jax.numpy as jnp
import numpy as np
from numpy.linalg import svd

# def unit_ball()
from scipy.stats import multivariate_normal


class unit_hypercube(object):
    def __init__(self, dim):
        self.dim = dim

    def rvs(self, size):
        return np.random.uniform(-1, 1, (size, self.dim))

    def logpdf(self, x):
        return np.where(np.abs(x) > 1, -np.inf, 0)

    def pdf(self, x):
        return np.where(np.abs(x) > 1, 0, 1)


class unit_hyperball(object):
    def __init__(self, dim, scale=1.0, loc=0.0):
        self.dim = dim
        self.scale = scale
        self.loc = loc

    def rvs(self, size):
        x = np.random.randn(size, self.dim)
        r = np.random.rand(size) ** (1 / self.dim)
        return (
            x / np.linalg.norm(x, axis=1)[:, None] * r[:, None] * self.scale + self.loc
        )

    def logpdf(self, x):
        r = np.linalg.norm(x, axis=1)
        return np.where(r > 1, -np.inf, (self.dim - 1) * np.log(r))

    def pdf(self, x):
        r = np.linalg.norm(x, axis=1)
        return np.where(r > 1, 0, (self.dim - 1) / 2 * r ** (self.dim - 1))


class ellipse(object):
    def __init__(self, loc, cov):
        self.dim = loc.shape[0]
        self.mean = loc
        self.cov = cov
        self.cov_inv = jnp.linalg.inv(cov.T)

    def _sample_n_and_log_prob(self, seed, n):
        x, pi = distrax.MultivariateNormalDiag(
            jnp.zeros(self.dim), jnp.ones(self.dim)
        )._sample_n_and_log_prob(seed, n)
        # x, pi = distrax.MultivariateNormalFullCovariance(
        #     self.mean, self.cov
        # )._sample_n_and_log_prob(seed, n)
        r, pi = distrax.Uniform(0, 1)._sample_n_and_log_prob(seed, n)
        r = r ** (1 / self.dim)
        x = (
            x / jnp.linalg.norm(x, axis=1)[:, None] * r[:, None]
        ) @ self.cov + self.mean
        return x, pi

    def _sample_n(self, seed, n):
        return self._sample_n_and_log_prob(seed, n)[0]

    def log_prob(self, x):
        return jnp.zeros(x.shape[0])

    def rvs(self, size):
        x = np.random.randn(size, self.dim)
        r = np.random.rand(size) ** (1 / self.dim)
        return (x / np.linalg.norm(x, axis=1)[:, None] * r[:, None]) @ np.linalg.inv(
            self.cov.T
        ) + self.mean

    def logpdf(self, x):
        return multivariate_normal(self.mean, self.cov).logpdf(x)
