import os

import anesthetic as ns
import matplotlib.pyplot as plt
import numpy as np

# import jax.random as rng
# from lsbi.model import LinearModel, MixtureModel
from scipy.optimize import rosen
from scipy.stats import multivariate_normal, norm, uniform

from fusions.cfm import CFM
from fusions.diffusion import Diffusion
from fusions.integrate import NestedDiffusion, SequentialDiffusion

os.makedirs("plots", exist_ok=True)

dims = 2


class likelihood(object):
    def logpdf(self, x):
        logl = -np.sum(
            100.0 * (x[..., 1:] - x[..., :-1] ** 2.0) ** 2.0 + (1 - x[..., :-1]) ** 2.0,
            axis=-1,
        )
        # mask = (x > 5).any(axis=-1)
        # logl[mask] = np.inf
        return logl

    def __call__(self, x):
        return np.log(
            np.sum(
                100.0 * (x[..., 1:] - x[..., :-1] ** 2.0) ** 2.0
                + (1 - x[..., :-1]) ** 2.0,
                axis=-1,
            )
        )


from lsbi.model import LinearModel

d = 100
t = dims
np.random.seed(1)

# M is the linear model matrix, C is the data covariance (error bars on data), this model encodes a data generating process of
# D = M @ theta + N(0, C)
# lm = LinearModel(
#     M=np.random.randn(d, t),
#     C=np.eye(d),
# )
# true_theta, true_data = np.split(lm.joint().rvs(), [t])

# class likelihood(object):
#     def logpdf(self, x):
#         return lm.likelihood(x).logpdf(true_data)

#     def __call__(self, x):
#         return lm.likelihood(x).logpdf(true_data)

from distrax import MultivariateNormalDiag, Normal, Uniform

prior = Uniform(low=-5.0 * np.ones(dims), high=5.0 * np.ones(dims))
# prior = MultivariateNormalDiag(
#             loc=np.zeros(dims), scale_diag=np.ones(dims)
#         )
# class prior(object):
#     dim: int = dims

#     def rvs(self, size):
#         return np.random.rand(size, dims) * 10 - 5

#     def logpdf(self, x):
#         return np.ones(x.shape[0]) * 1.0


from fusions.network import ScoreApprox

network = ScoreApprox(n_initial=128, n_hidden=32, n_layers=3, n_fourier_features=4)
model = CFM

# model = Diffusion
diffuser = NestedDiffusion(prior=prior, likelihood=likelihood(), model=model)
diffuser.settings.target_eff = 1.0
diffuser.settings.epoch_factor = 10
diffuser.settings.n = 1000
diffuser.settings.noise = 1e-3
diffuser.settings.prior_boost = 2
diffuser.settings.eps = 1e-2
diffuser.settings.batch_size = 500
diffuser.settings.restart = True
diffuser.settings.lr = 1e-2
diffuser.score_model = network
import jax

diffuser.rng = jax.random.PRNGKey(10)
# diffuser.run(steps=10, n=500)
diffuser.run()
# print(lm.evidence().logpdf(true_data))
samples = diffuser.samples()
diffuser.write("diffusion")
# print(f"analytic: {logz:.2f}")
zs = samples.logZ(50)
print(f"numerical estimation: {zs.mean():.2f} +- {zs.std():.2f}")
print(samples.logZ())
print(samples.logZ(30).std())

total_samples = len(samples.compress(200))
print(f"total samples: {total_samples}")
size = total_samples * 2
theta = np.asarray(prior._sample_n(jax.random.PRNGKey(0), size))
# P = TargetModel.posterior(data).rvs(size * 3)
a = ns.MCMCSamples(theta).plot_2d(np.arange(dims))

f, a = ns.make_2d_axes(np.arange(dims))
# ns.MCMCSamples(theta).plot_2d(a, alpha=0.3, label="Prior")
# ns.MCMCSamples(P).plot_2d(a, alpha=0.3, label="Analytic")
samples.plot_2d(a, label="NestedDiffusion")
plt.legend()
f.savefig("plots/ns.pdf")
