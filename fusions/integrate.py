import logging

from fusions.cfm import CFM
from fusions.diffusion import Diffusion

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pickle import dump, load

import anesthetic.termination as term
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from anesthetic import MCMCSamples, NestedSamples, make_2d_axes, read_chains
from anesthetic.utils import compress_weights, neff
from blackjax.smc.tuning.from_particles import (
    particles_covariance_matrix,
    particles_means,
    particles_stds,
)
from distrax import (
    MultivariateNormalDiag,
    MultivariateNormalFullCovariance,
    Normal,
    Uniform,
)
from jax import random
from jax.lax import scan, while_loop

# from anesthetic.read.hdf import read_hdf, write_hdf
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from tqdm import tqdm

from fusions.model import Model
from fusions.utils import ellipse, unit_hyperball, unit_hypercube


@dataclass
class Point:
    x: np.ndarray
    # latent_x: np.ndarray
    logl: float
    logl_birth: float
    logl_pi: float = field(default=0.0)


@dataclass
class Stats:
    nlive: int = field(default=0)
    nlike: int = field(default=0)
    ndead: int = field(default=0)
    logz: float = field(default=-1e30)
    logz_err: float = field(default=1)
    logX: float = field(default=1)

    def __repr__(self):
        return (
            f"Stats(\n"
            f"  nlive: {self.nlive},\n"
            f"  nlike: {self.nlike},\n"
            f"  ndead: {self.ndead},\n"
            f"  logz: {self.logz},\n"
            f"  logz_err: {self.logz_err},\n"
            f"  logX: {self.logX}\n"
            f")"
        )


@dataclass
class Settings:
    """Settings for the integrator.

    Args:
        n (int, optional): Number of samples to draw. Defaults to 500.
        target_eff (float, optional): Target efficiency. Defaults to 0.1.
        steps (int, optional): Number of steps to take. Defaults to 20.
        prior_boost (int, optional): Number of samples to draw from the prior. Defaults to 5.
        eps (float, optional): Tolerance for the termination criterion. Defaults to 1e-3.
        batch_size (float, optional): Batch size for training the diffusion. Defaults to 0.25.
        epoch_factor (int, optional): Factor to multiply the number of epochs by. Defaults to 1.
        restart (bool, optional): Whether to restart the training. Defaults to False.
        noise (float, optional): Noise to add to the training. Defaults to 1e-3.
        efficiency (float, optional): Efficiency. Defaults to 1 / np.e.
        logzero (float, optional): Value to use for log zero. Defaults to -1e30.
    """

    n: int = 500
    target_eff: float = 1.0
    steps: int = 20
    prior_boost: int = 5
    eps: float = 1e-3
    batch_size: float = 0.25
    epochs: int = 1
    restart: bool = True
    noise: float = 1e-3
    resume: bool = False
    dirname: str = "fusions_samples"
    lr: float = 1e-3
    gamma: float = 0.9
    # efficiency: float = 1 / np.e
    # logzero: float = -1e30

    def __repr__(self):
        return (
            f"Settings(\n"
            f"  n: {self.n},\n"
            f"  target_eff: {self.target_eff},\n"
            f"  steps: {self.steps},\n"
            f"  prior_boost: {self.prior_boost},\n"
            f"  eps: {self.eps},\n"
            f"  batch_size: {self.batch_size},\n"
            f"  epoch_factor: {self.epochs},\n"
            f"  restart: {self.restart},\n"
            f"  noise: {self.noise},\n"
            f"  resume: {self.resume},\n"
            # f"  efficiency: {self.efficiency},\n"
            f")"
        )


@dataclass
class Trace:
    diff: Point = field(default_factory=dict)
    live: Point = field(default_factory=dict)
    flow: Point = field(default_factory=dict)
    prior: Point = field(default_factory=dict)
    accepted_live: Point = field(default_factory=dict)
    iteration: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=dict)
    lr: list[float] = field(default_factory=dict)
    efficiency: list[float] = field(default_factory=list)


class Integrator(ABC):
    def __init__(self, prior, likelihood, **kwargs) -> None:
        self.prior = prior
        self.likelihood = likelihood
        self.logzero = kwargs.get("logzero", -np.inf)
        self.dead = []
        self.dists = []
        self.stats = Stats()
        self.settings = Settings()
        self.model = kwargs.get("model", CFM)
        # self.dim = prior.dim
        self.rng = kwargs.get("rng", random.PRNGKey(0))
        self.dim = prior.sample(seed=random.PRNGKey(0)).shape[0]
        self.trace = Trace()
        # self.prior = multivariate_normal(
        #     np.zeros(prior.dim), np.eye(prior.dim)
        # )
        self.latent = MultivariateNormalDiag(
            loc=np.zeros(self.dim), scale_diag=np.ones(self.dim)
        )
        # self.latent = Normal(loc=np.zeros(self.dim), scale=np.ones(self.dim))
        # self.latent = multivariate_normal(np.zeros(prior.dim), np.eye(prior.dim))
        # self.latent = unit_hyperball(prior.dim)
        # self.latent = self.prior

    def sample(self, n, dist, logl_birth=0.0, beta=1.0, **kwargs):
        if isinstance(dist, Model):
            boost = kwargs.get("boost", 1.0)

            def body(rng, xi):
                rng, sample_key = random.split(rng)
                x0, pi0 = dist.prior._sample_n_and_log_prob(sample_key, n)
                x1, j = dist.reverse_process(
                    x0,
                    dist._predict,
                    sample_key,
                    solution="exact",
                    # solution="approx",
                )
                pi1 = jnp.sum(self.prior.log_prob(x1), axis=-1)
                # pi0 = jnp.sum(pi0, axis=-1)
                logw = pi1 - pi0 + j
                return rng, (x1, logw)

            keys = random.split(self.rng, 5)

            _, (x, logw) = scan(body, self.rng, keys)
            x = jnp.concatenate(x, axis=0)
            logw = jnp.concatenate(logw, axis=0)

            p1, p0, j0 = kwargs.get("logw")

            oob_mask = jnp.isinf(logw)
            min_p_mask = logw < (p1 - p0 - j0).max()
            mask = jnp.logical_and(min_p_mask, ~oob_mask)
            # mask = ~oob_mask
            x = x[~oob_mask]
            logw = logw[~oob_mask]

            # unweighting
            # x=x[mask]
            # logw=logw[mask]

            # self.rng, compress_key = random.split(self.rng)
            # idx = (
            #    jnp.log(Uniform()._sample_n(compress_key, logw.shape[0]))
            #     <= logw - logw.max()
            # )
            # x = x[idx]
            # logw = logw[idx]

        else:
            self.rng, sample_key = random.split(self.rng)
            x, _ = dist._sample_n_and_log_prob(sample_key, n)
            logw = jnp.zeros(n)
            # w = jnp.sum(jnp.atleast_2d(pi), axis=-1)
            # w = pi

        # necessary_idx = min(n, x.shape[0])
        # self.rng, choice_key = random.split(self.rng)
        # idx = jnp.asarray(
        #     jnp.random.choice(x.shape[0], necessary_idx, replace=False)
        # )

        logl = self.likelihood.logpdf(x)
        l_idx = logl > logl_birth
        x = x[l_idx]
        logw = logw[l_idx]
        logl = logl[l_idx]
        # self.rng, compress_key = random.split(self.rng)

        # idx = (
        #     jnp.log(Uniform()._sample_n(compress_key, logw.shape[0]))
        #     <= logw - logw.max()
        # )
        # idx = compress_weights(jnp.exp(w.flatten()), ncompress="equal")
        # idx = compress_weights(
        #     jnp.exp(jnp.ones_like(w).flatten()), ncompress="equal"
        # )
        # idx = np.asarray(idx, dtype=bool)

        # idx = compress_weights(
        #     jnp.exp(jnp.ones_like(w).flatten()), ncompress="equal"
        # )
        # idx = np.asarray(idx, dtype=bool)

        # logl = self.likelihood.logpdf(x[idx])

        # l_idx = logl > logl_birth
        # x = x[idx][l_idx]
        # logl = logl[l_idx]
        # logl_birth = np.ones_like(logl) * logl_birth
        # x = x[idx]

        # x = x[idx]
        # logl = logl[idx]
        # logl_birth = np.ones_like(logl) * logl_birth

        # logl = self.likelihood.logpdf(x)
        # l_idx = logl > logl_birth
        # x = x[l_idx]
        # logl = logl[l_idx]
        # logl_birth = np.ones_like(logl) * logl_birth

        # necessary_idx = min(n, x.shape[0])
        # self.rng, choice_key = random.split(self.rng)
        # idx = random.choice(choice_key, jnp.arange(x.shape[0]), shape=(necessary_idx,), replace=False)
        # x = x[idx]
        # logl = logl[idx]
        # logl_birth = logl_birth[idx]

        self.stats.nlike += l_idx.shape[0]
        logl_birth = np.ones_like(logl) * logl_birth
        points = [
            Point(
                np.asarray(x[i]),
                logl[i],
                logl_birth[i],
            )
            for i in range(logl.shape[0])
        ]
        return points

    # def sample(self, n, dist, logl_birth=0.0, beta=1.0):

    #     if isinstance(dist, Model):
    #         self.rng, sample_key = random.split(self.rng)
    #         x0, pi0 = dist.prior._sample_n_and_log_prob(sample_key, n)
    #         # pi0 = jnp.sum(pi0, axis=-1)
    #         # logging.log(logging.INFO, f"Sampling {n} points")
    #         # latent_x = dist.prior.rvs(n)
    #         # x, j = dist.predict(latent_x, jac=True, solution="exact")
    #         self.rng, step_rng = random.split(self.rng)
    #         x1, j = dist.reverse_process(
    #             x0,
    #             dist._predict,
    #             step_rng,
    #             solution="exact",
    #         )

    #         pi1 = jnp.sum(self.prior.log_prob(x1), axis=-1)
    #         w = pi1 - j
    #         x = x1
    #     else:
    #         self.rng, sample_key = random.split(self.rng)
    #         x, pi = dist._sample_n_and_log_prob(sample_key, n)
    #         w = jnp.ones(n)
    #     #     # x = np.asarray(dist.rvs(n))
    #     #     w = np.ones(n)
    #     #     latent_x = x
    #     return np.asarray(x), np.asarray(w)

    def stash(self, points, n, drop=False):
        live = sorted(points, key=lambda lp: lp.logl, reverse=True)
        if not drop:
            self.dead += live[n:]
        contour = live[n].logl
        live = live[:n]
        return live, contour

    @abstractmethod
    def run(self, **kwargs):
        pass

    @abstractmethod
    def update_stats(self, live, n):
        pass

    @abstractmethod
    def points_to_samples(self, points):
        pass

    def samples(self):
        return self.points_to_samples(self.dead)

    def importance_integrate(self, dist, n=1000):
        points = dist.rvs(n)
        likelihood = self.likelihood.logpdf(points)
        prior = self.prior.logpdf(points)
        return logsumexp(likelihood + prior) - np.log(n)

    def write(self, filename):
        os.makedirs(self.settings.dirname, exist_ok=True)
        self.points_to_samples(self.dead).to_csv(
            os.path.join(self.settings.dirname, filename) + ".csv"
        )

    def read(self, filename):
        self.dead = read_chains(filename)

    def write_trace(self, filename):
        os.makedirs(self.settings.dirname, exist_ok=True)
        dump(
            self.trace,
            open(os.path.join(self.settings.dirname, filename) + ".pkl", "wb"),
        )


class NestedDiffusion(Integrator):
    def sample_constrained(self, n, dist, constraint, efficiency=1.0, **kwargs):
        pi = self.sample(n, dist, constraint, **kwargs)
        eff = len(pi) / n
        return True, eff, pi
        # success = []
        # trials = 0
        # while len(success) < n:
        #     batch_success = []
        #     pi = self.sample(n, dist, constraint, **kwargs)
        #     batch_success += pi  # [p for p in pi if p.logl > constraint]
        #     success += batch_success
        #     trials += int(n / efficiency)
        # eff = np.round(len(success) / trials, decimals=3)
        # return eff > efficiency, eff, success

    # def sample_constrained(
    #     self, n, dist, constraint, efficiency=0.1, **kwargs
    # ):
    #     success = 0
    #     x = []
    #     w = []
    #     l = []
    #     pbar = tqdm(total=n)
    #     while success < n:
    #         xi, wi = self.sample(int(n/efficiency), dist, **kwargs)
    #         x.append(xi)
    #         w.append(wi)
    #         # l.append(li)
    #         coords = np.concatenate(x)
    #         weights = np.concatenate(w)
    #         like = np.concatenate(l)
    #         # l_mask = like > constraint
    #         # l_mask = np.asarray(l_mask, dtype="bool")
    #         idx = compress_weights(
    #             jnp.exp(weights.flatten()), ncompress="equal"
    #         )
    #         # idx = compress_weights(weights[l_mask].flatten())

    #         # idx = np.ones(l_mask.sum())
    #         # w_mask = np.asarray(idx, dtype=bool)

    #         logger.info(f"Sampled {n} new points, ess is: {idx.sum()}")
    #         pbar.update(idx.sum() - success)
    #         success = idx.sum()
    #         pbar.close()
    #     logl_birth = np.ones(success) * constraint

    #     points = [
    #         # Point(
    #         #     coords[l_mask][w_mask][i],
    #         #     like[l_mask][w_mask][i],
    #         #     logl_birth[i],
    #         #     # logl_pi=log_pi[idx][i],
    #         # )
    #         Point(
    #             np.repeat(coords[l_mask], idx, axis=0)[i],
    #             np.repeat(like[l_mask], idx, axis=0)[i],
    #             logl_birth[i],
    #             # logl_pi=log_pi[idx][i],
    #         )
    #         for i in range(success)
    #     ]
    #     return points, l_mask.mean(), idx.mean()

    def train_diffuser(self, dist, points, prior_samples=None):
        dist.train(
            np.asarray([yi.x for yi in points]),
            epochs=int(self.settings.epochs),
            # batch_size=int(len(points) * self.settings.batch_size),
            batch_size=self.settings.batch_size,
            lr=self.settings.lr,
            restart=self.settings.restart,
            # noise=self.settings.noise,
            prior_samples=prior_samples,
        )
        return dist

    def run(self):
        n = self.settings.n
        live = self.sample(
            n * 2 * self.settings.prior_boost, self.prior, self.logzero
        )  #
        # live, _, _ = self.sample_constrained(
        #     n * self.settings.prior_boost, self.prior, self.logzero
        # )
        step = 0
        logger.info("Done sampling prior")
        # live, contour = self.stash(live, n // 2, drop=False)
        live, contour = self.stash(live, n, drop=False)
        self.dist = self.prior
        self.update_stats(live, n)
        logger.info(f"{self.stats}")
        self.dist = self.model(self.latent, noise=self.settings.noise)
        eff = 1 / self.dim
        while not self.points_to_samples(live + self.dead).terminated(
            criterion="logZ", eps=self.settings.eps
        ):
            # xi, ci = np.random.choice(len(live), (2, len(live) // 2), replace=False)
            print(len(live))
            self.live = live
            live_as_array = np.asarray([xi.x for xi in live])
            # ellipse(particles_means(live_as_array), particles_covariance_matrix(live_as_array))
            # ellipse(particles_means(live_as_array), particles_covariance_matrix(live_as_array))._sample_n_and_log_prob(self.rng,100)

            self.dist = self.model(
                MultivariateNormalFullCovariance(
                    particles_means(live_as_array),
                    particles_covariance_matrix(live_as_array),
                ),
                noise=particles_stds(live_as_array) * self.settings.noise,
                # noise = self.settings.noise
            )
            # self.dist = self.model(self.prior,noise=particles_stds(live_as_array) * self.settings.noise,)
            # self.dist = self.model(
            #     self.prior,
            #     noise=particles_stds(live_as_array) * self.settings.noise,
            #     # noise = self.settings.noise
            # )

            # self.dist = self.model(ellipse(particles_means(live_as_array),2* particles_covariance_matrix(live_as_array)), noise=particles_stds(live_as_array) * self.settings.noise)
            print(particles_covariance_matrix(live_as_array))
            print(particles_means(live_as_array))

            # self.dist = self.model(
            #     MultivariateNormalFullCovariance(
            #         jnp.zeros(self.dim),
            #         jnp.eye(self.dim),
            #     ),
            #     # noise=particles_stds(live_as_array) * self.settings.noise,
            #     noise = self.settings.noise
            # )
            self.dist = self.train_diffuser(self.dist, live)

            live_as_array = np.asarray([xi.x for xi in live])
            x0, j0 = self.dist.reverse_process(
                live_as_array,
                self.dist._predict,
                self.rng,
                solution="exact",
                t0=1.0,
                t=0.0,
                dt0=-1e-3,
            )
            p0 = self.dist.prior.log_prob(x0)
            p1 = jnp.sum(self.prior.log_prob(live_as_array), axis=-1)
            # pseudo_importance_weight = p1 - p0 + j0

            # x0,j0 = self.dist.reverse_process(
            #     np.asarray([xi.x for xi in live]),
            #     self.dist._predict,
            #     self.rng,
            #     solution="exact",
            #     t0 = 1.0,
            #     t = 0.0,
            #     dt0 = -1e-3,
            # )

            # x = self.dist.rvs(len(live))
            # self.dist.calibrate(
            #     x,
            #     jnp.asarray([xi.x for xi in live]),
            #     batch_size=self.settings.batch_size,
            #     lr=self.settings.lr,
            #     epochs=int(50 * self.settings.epoch_factor),
            # )

            self.trace.losses[step] = self.dist.trace.losses
            self.trace.lr[step] = self.dist.trace.lr

            success, eff, points = self.sample_constrained(
                n,
                self.dist,
                contour,
                efficiency=self.settings.target_eff,
                logw=(p1, p0, j0),
                boost=1 / eff,
                # pmin = self.dist.prior.log_prob(x0).min()
            )

            # points, eff, _ = self.sample_constrained(
            #     n // 2,
            #     self.dist,
            #     contour,
            #     efficiency=self.settings.target_eff,
            # )
            logger.info(f"Efficiency at: {eff}, using previous diffusion")

            self.trace.live[step] = live
            self.trace.accepted_live[step] = points
            # self.trace.flow[step] = x
            self.trace.prior[step] = self.dist.prior._sample_n(self.rng, n)
            x1, j = self.dist.reverse_process(
                self.trace.prior[step],
                self.dist._predict,
                self.rng,
                solution="exact",
                # solution="approx",
            )
            self.trace.flow[step] = x1
            self.trace.diff[step] = j
            live = live + points
            # live, contour = self.stash(live, n // 2, drop=False)
            live, contour = self.stash(live, n, drop=False)

            self.update_stats(live, n)
            logger.info(f"{self.stats}")
            step += 1
            self.trace.iteration.append(step)
            self.trace.efficiency.append(eff)
            self.write_trace("trace")

            logger.info(f"Step {step} complete")

        self.stash(live, -len(live))
        self.write_trace("trace")
        logger.info(f"Final stats: {self.stats}")

    def update_stats(self, live, n):
        running_samples = self.points_to_samples(self.dead)  # +live
        self.stats.ndead = len(self.dead)
        lZs = running_samples.logZ(100)
        i_live = running_samples.live_points().index
        self.stats.logX = np.exp(running_samples.logX().iloc[i_live[0]])
        # self.stats.logX = running_samples.critical_ratio()
        # running_samples.terminated()
        self.stats.logz = lZs.mean()
        self.stats.logz_err = lZs.std()

    def points_to_samples(self, points):
        return NestedSamples(
            data=[p.x for p in points],
            logL=[p.logl for p in points],
            logL_birth=[p.logl_birth for p in points],
        )

    def points_to_samples_importance(self, points, weights):
        return MCMCSamples(
            data=[p.x for p in points],
            weights=weights,
            # weights=np.exp([p.logl_pi for p in points]),
        )

    def samples(self):
        return self.points_to_samples(self.dead)


# class SequentialDiffusion(Integrator):
#     beta_min = 0.00001
#     beta_max = 1.0

#     def run(self, n=1000, steps=10, schedule=np.linspace, **kwargs):
#         target_eff = kwargs.get("efficiency", 0.5)

#         betas = schedule(self.beta_min, self.beta_max, steps)
#         self.dist = self.prior
#         diffuser = self.model(self.prior)

#         for beta_i in betas:
#             live = self.sample(n * 2, self.dist, beta=beta_i)
#             frame = self.points_to_samples(live)

#             ess = len(frame.compress())
#             while ess < n:
#                 live += self.sample(n, self.dist, beta=beta_i)
#                 frame = self.points_to_samples(live)
#                 ess = len(frame.compress())
#                 logger.info(
#                     f"Efficiency at: {ess/len(live)}, using previous diffusion"
#                 )
#             logger.info(f"Met ess criteria, training new diffusion")
#             diffuser = self.model(self.prior)
#             diffuser.train(
#                 np.asarray(self.points_to_samples(live).compress()),
#                 n_epochs=n,
#                 batch_size=n,
#                 lr=1e-3,
#             )

#             self.dist = diffuser
#             self.dead += live
#             self.update_stats(live, n)
#             logger.info(f"{self.stats}")

#         self.dead = self.sample(n * 4, self.dist)

#     def update_stats(self, live, n):
#         running_samples = self.points_to_samples(self.dead)
#         self.stats.ndead = len(running_samples.compress())
#         self.stats.logz = np.log(running_samples.get_weights().mean())

#     def points_to_samples(self, points):
#         if not points:
#             return MCMCSamples(data=[], weights=[])
#         else:
#             logls = np.asarray([p.logl for p in points])
#             logls -= logls.max()
#             return MCMCSamples(
#                 data=[p.x for p in points], weights=np.exp(logls)
#             )
