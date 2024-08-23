from abc import ABC, abstractmethod

import numpy as np
from jax import numpy as jnp
from jax import random

# from ott.geometry import pointcloud
# from ott.solvers import linear


class OTBase(ABC):
    def __init__(self, x0, x1):
        self.x0 = np.atleast_2d(x0)
        self.x1 = np.atleast_2d(x1)
        self.rng = np.random.default_rng(0)

    @abstractmethod
    def sample(self, rng=None, batch_size=128):
        pass


class NullOT(OTBase):
    def __init__(self, x0, x1):
        super().__init__(x0, x1)
        self.rng = np.random.default_rng(0)

    def sample(self, batch_size=128, *args):
        idx = self.rng.choice(self.x0.shape[0], size=(batch_size, 2), replace=True)
        return idx[..., 0], idx[..., 1]


class PriorExtendedNullOT(OTBase):
    def __init__(self, x0, x1):
        super().__init__(x0, x1)
        self.rng = np.random.default_rng(0)

    def sample(self, batch_size=128, *args):
        idx = self.rng.choice(self.x0.shape[0], size=(batch_size), replace=True)
        idx_p = self.rng.choice(self.x1.shape[0], size=(batch_size), replace=True)
        return idx, idx_p


class FullOT(OTBase):
    def __init__(self, x0, x1):
        super().__init__(x0, x1)
        geom = pointcloud.PointCloud(self.x0, self.x1)
        self.M = np.asarray(linear.solve(geom).matrix.flatten())
        self.M = self.M / self.M.sum()

    def sample(self, batch_size=128, *args):
        # gumbel_noise = random.gumbel(rng, shape=(batch_size,) + self.M.shape)
        # idx = random.categorical(rng, self.M.flatten(), shape=(batch_size,))
        # idx = jnp.divmod(idx, batch_size)
        # return idx[0], idx[1]
        idx = self.rng.choice(self.M.shape[0], size=batch_size, p=self.M, replace=False)
        idx = np.divmod(idx, batch_size)
        return idx[0], idx[1]


# class FastFullOT(OTBase):
#     def __init__(self, x0, x1):
#         super().__init__(x0, x1)
#         geom = pointcloud.PointCloud(self.x0, self.x1)
#         self.M = np.asarray(linear.solve(geom).matrix)
#         self.M = self.M / self.M.sum()

#     def sample_matrix(self, batch_size=128, *args):
#         idx_1 = self.rng.choice(self.M.shape[0], size=batch_size, replace=False)
#         idx_0 = np.argmax(self.M[..., idx_1], axis=0)
#         return idx_0, idx_1
