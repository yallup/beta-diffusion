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
from distrax import MultivariateNormalDiag, Normal, Uniform
from jax import random

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
    epoch_factor: int = 1
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
            f"  epoch_factor: {self.epoch_factor},\n"
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

    def sample(self, n, dist, logl_birth=0.0, beta=1.0):
        if isinstance(dist, Model):
            self.rng, sample_key = random.split(self.rng)
            x0, pi0 = dist.prior._sample_n_and_log_prob(sample_key, n)
            # pi0 = jnp.sum(pi0, axis=-1)

            self.rng, step_rng = random.split(self.rng)
            # x1, j = dist.guided_reverse_process(
            #     x0,
            #     dist._predict,
            #     dist._calibrator_score,
            #     step_rng,
            #     solution="exact",
            # )
            x1, j = dist.reverse_process(
                x0,
                dist._predict,
                step_rng,
                solution="exact",
            )
            logl = self.likelihood.logpdf(x1) * beta
            # mask = logl > logl_birth
            pi1 = jnp.sum(self.prior.log_prob(x1), axis=-1)
            # pi1 = self.prior.log_prob(x1)
            # pi = j.squeeze() - pi0 - pi1
            pi = pi1 - pi0 + j

            mask = ~jnp.isinf(pi)  # & (dist._calibrate_predict(x1).squeeze() < 0.5)

            x = x1[mask]
            w = pi[mask]

        else:
            self.rng, sample_key = random.split(self.rng)
            x, pi = dist._sample_n_and_log_prob(sample_key, n)
            w = jnp.ones(n)
            # w = jnp.sum(jnp.atleast_2d(pi), axis=-1)
            # w = pi

        logl = self.likelihood.logpdf(x)
        l_idx = logl > logl_birth
        x = x[l_idx]
        w = w[l_idx]
        logl = logl[l_idx]

        # self.rng, compress_key = random.split(self.rng)
        # idx = jnp.log(Uniform()._sample_n(compress_key, w.shape[0])) <= w - w.max()
        idx = compress_weights(jnp.exp(w.flatten()), ncompress="equal")
        idx = np.asarray(idx, dtype=bool)
        logl = self.likelihood.logpdf(x) * beta
        self.stats.nlike += l_idx.shape[0]
        logl_birth = np.ones_like(logl) * logl_birth
        points = [
            Point(
                np.asarray(x[idx][i]),
                logl[idx][i],
                logl_birth[idx][i],
            )
            for i in range(idx.sum())
        ]
        return points

    # def sample(self, n, dist, logl_birth=0.0, beta=1.0):

    #     if isinstance(dist, Model):
    #         self.rng, sample_key = random.split(self.rng)
    #         x0, pi0 = dist.prior._sample_n_and_log_prob(sample_key, n)
    #         pi0 = jnp.sum(pi0, axis=-1)
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
    #         w = j.squeeze() - pi0 + pi1
    #         # latent_x = np.asarray(latent_x)
    #         # latent_pi = dist.prior.logpdf(latent_x)
    #         # x = np.asarray(x).squeeze()
    #         x = x1
    #         pi = jnp.exp(j.squeeze() - pi0 + pi1)
    #         # theta =np.asarray([xi.x for xi in self.live])
    #         # # a = MCMCSamples(theta.squeeze()).plot_2d()
    #         # psi, psi_j = dist.reverse_process(
    #         #     theta,
    #         #     dist._predict,
    #         #     step_rng,
    #         #     solution="exact",
    #         #     t0=1.0,t1=0.0, dt0=-1e-2
    #         # )
    #         # a = MCMCSamples(psi.squeeze()).plot_2d()
    #         # MCMCSamples(latent_x).plot_2d(a)
    #         # plt.savefig("plots/latent.pdf")
    #         # theta =np.asarray([xi.x for xi in self.live])
    #         # w = np.exp(latent_pi)
    #         # w = np.exp(j.squeeze())
    #         # w = np.ones_like(j)
    #     else:
    #         self.rng, sample_key = random.split(self.rng)
    #         x, pi = dist._sample_n_and_log_prob(sample_key, n)
    #         pi = jnp.sum(pi, axis=-1)
    #     #     # x = np.asarray(dist.rvs(n))
    #     #     w = np.ones(n)
    #     #     latent_x = x
    #     return np.asarray(x), np.asarray(pi)

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
        success = []
        trials = 0
        while len(success) < n:
            batch_success = []
            pi = self.sample(int(n / efficiency), dist, constraint, **kwargs)
            batch_success += pi  # [p for p in pi if p.logl > constraint]
            success += batch_success
            trials += int(n / efficiency)
        eff = np.round(len(success) / trials, decimals=3)
        return eff > efficiency, eff, success

    # def sample_constrained(
    #     self, n, dist, constraint, efficiency=0.1, **kwargs
    # ):
    #     success = 0
    #     x = []
    #     w = []
    #     l = []
    #     pbar = tqdm(total=n)
    #     while success < n:
    #         xi, wi = self.sample(n, dist, **kwargs)
    #         self.stats.nlike += n
    #         li = self.likelihood.logpdf(xi)
    #         x.append(xi)
    #         w.append(wi)
    #         l.append(li)
    #         coords = np.concatenate(x)
    #         weights = np.concatenate(w)
    #         like = np.concatenate(l)
    #         l_mask = like > constraint
    #         l_mask = np.asarray(l_mask, dtype="bool")
    #         idx = compress_weights(
    #             weights[l_mask].flatten(), ncompress="equal"
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
            epochs=int(50 * self.settings.epoch_factor),
            # batch_size=int(len(points) * self.settings.batch_size),
            batch_size=self.settings.batch_size,
            lr=self.settings.lr,
            restart=self.settings.restart,
            noise=self.settings.noise,
            prior_samples=prior_samples,
        )
        return dist

    def run(self):
        n = self.settings.n
        live = self.sample(n * self.settings.prior_boost, self.prior, self.logzero)  #
        # live, _, _ = self.sample_constrained(
        #     n * self.settings.prior_boost, self.prior, self.logzero
        # )
        step = 0
        logger.info("Done sampling prior")
        live, contour = self.stash(live, n // 2, drop=False)
        self.dist = self.prior
        self.update_stats(live, n)
        logger.info(f"{self.stats}")
        self.dist = self.model(self.latent, noise=self.settings.noise)

        while not self.points_to_samples(live + self.dead).terminated(
            criterion="logZ", eps=self.settings.eps
        ):
            # xi, ci = np.random.choice(len(live), (2, len(live) // 2), replace=False)
            self.live = live
            self.dist = self.train_diffuser(self.dist, live)
            x = self.dist.rvs(len(live))
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
                n // 2,
                self.dist,
                contour,
                efficiency=self.settings.target_eff,
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
            self.trace.flow[step] = x
            self.trace.prior[step] = self.dist.prior._sample_n(self.rng, n)
            live = live + points
            live, contour = self.stash(live, n // 2, drop=False)

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


class SequentialDiffusion(Integrator):
    beta_min = 0.00001
    beta_max = 1.0

    def run(self, n=1000, steps=10, schedule=np.linspace, **kwargs):
        target_eff = kwargs.get("efficiency", 0.5)

        betas = schedule(self.beta_min, self.beta_max, steps)
        self.dist = self.prior
        diffuser = self.model(self.prior)

        for beta_i in betas:
            live = self.sample(n * 2, self.dist, beta=beta_i)
            frame = self.points_to_samples(live)

            ess = len(frame.compress())
            while ess < n:
                live += self.sample(n, self.dist, beta=beta_i)
                frame = self.points_to_samples(live)
                ess = len(frame.compress())
                logger.info(f"Efficiency at: {ess/len(live)}, using previous diffusion")
            logger.info(f"Met ess criteria, training new diffusion")
            diffuser = self.model(self.prior)
            diffuser.train(
                np.asarray(self.points_to_samples(live).compress()),
                n_epochs=n,
                batch_size=n,
                lr=1e-3,
            )

            self.dist = diffuser
            self.dead += live
            self.update_stats(live, n)
            logger.info(f"{self.stats}")

        self.dead = self.sample(n * 4, self.dist)

    def update_stats(self, live, n):
        running_samples = self.points_to_samples(self.dead)
        self.stats.ndead = len(running_samples.compress())
        self.stats.logz = np.log(running_samples.get_weights().mean())

    def points_to_samples(self, points):
        if not points:
            return MCMCSamples(data=[], weights=[])
        else:
            logls = np.asarray([p.logl for p in points])
            logls -= logls.max()
            return MCMCSamples(data=[p.x for p in points], weights=np.exp(logls))
