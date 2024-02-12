import math
from functools import partial

import diffrax as dfx
from diffrax.saveat import SaveAt

import jax
import jax.numpy as jnp
import jax.random as random
from fusions.model import Model
from jax import grad, jit, pmap, vjp, vmap


def normal_log_likelihood(y):
    return -0.5 * (y.size * math.log(2 * math.pi) + jnp.sum(y**2))


# def approx_logp_wrapper(t, y, args):
#     # y, _ = y
#     *args, eps, func = args
#     fn = lambda y: func(t, y, args)
#     f, vjp_fn = vjp(fn, y)
#     (eps_dfdy,) = vjp_fn(eps)
#     logp = jnp.sum(eps_dfdy * eps)
#     return f, logp


class CFM(Model):
    """Continuous Flows Model."""

    @partial(jit, static_argnums=[0, 2])
    def reverse_process(self, initial_samples, score, rng):
        """Run the reverse ODE.

        Args:
            initial_samples (jnp.ndarray): Samples to run the model on.
            score (callable): Score function.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Samples from the posterior distribution. and the history of the process.
        """
        t0, t1, dt0 = 0.0, 1.0, 1e-3
        ts = jnp.linspace(t0, t1, 100)

        def f(x):
            # return score(x, jnp.atleast_1d(t))
            def score_args(ti, xi, args):
                return score(xi, jnp.atleast_1d(ti))

            term = dfx.ODETerm(score_args)
            # solver = dfx.Heun()
            solver = dfx.Dopri5()
            sol = dfx.diffeqsolve(
                term, solver, t0, t1, dt0, x, saveat=SaveAt(t1=True, ts=ts)
            )
            return sol.ys

        yt = vmap(f)(initial_samples)
        return yt[:, -1, :], jnp.moveaxis(yt, 0, 1)

    @partial(jit, static_argnums=[0])
    def loss(self, params, batch, batch_prior, batch_stats, rng):
        """Loss function for training the CFM score."""
        sigma_noise = 1e-3
        rng, step_rng = random.split(rng)
        N_batch = batch.shape[0]

        t = random.uniform(step_rng, (N_batch, 1))
        x0 = batch_prior
        x1 = batch
        noise = random.normal(step_rng, (N_batch, self.ndims))
        psi_0 = t * batch + (1 - t) * batch_prior + sigma_noise * noise
        output, updates = self.state.apply_fn(
            {"params": params, "batch_stats": batch_stats},
            psi_0,
            t,
            train=True,
            mutable=["batch_stats"],
        )
        psi = x1 - x0
        loss = jnp.mean((output - psi) ** 2)
        return loss, updates


class VPCFM(CFM):
    """Continuous Flows Model with Volume preservation enforced."""

    @partial(jit, static_argnums=[0, 2])
    def reverse_process(self, initial_samples, score, rng):
        """Run the reverse ODE.

        Args:
            initial_samples (jnp.ndarray): Samples to run the model on.
            score (callable): Score function.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Samples from the posterior distribution. and the history of the process.
        """
        t0, t1, dt0 = 0.0, 1.0, 1e-3
        ts = jnp.linspace(t0, t1, 100)

        def f(x):
            # return score(x, jnp.atleast_1d(t))
            def score_args(ti, xi, args):
                return score(xi, jnp.atleast_1d(ti))

            term = dfx.ODETerm(score_args)
            # solver = dfx.Heun()
            solver = dfx.Dopri5()
            sol = dfx.diffeqsolve(
                term, solver, t0, t1, dt0, x, saveat=SaveAt(t1=True, ts=ts)
            )
            return sol.ys

        yt = vmap(f)(initial_samples)
        return yt[:, -1, :], jnp.moveaxis(yt, 0, 1)

    @partial(jit, static_argnums=[0, 2])
    def jac(self, initial_samples, score, rng):
        """Run the reverse ODE and track the jacobian.

        Args:
            initial_samples (jnp.ndarray): Samples to run the model on.
            score (callable): Score function.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Samples from the posterior distribution. and the history of the process.
        """
        t0, t1, dt0 = 0.0, 1.0, 1e-3
        ts = jnp.linspace(t0, t1, 100)

        def f(x):
            # return score(x, jnp.atleast_1d(t))
            # def approx_logp_wrapper(ti, xi, args):
            #     *args, eps, func = args
            #     fn = lambda yi: score(yi, jnp.atleast_1d(ti))
            #     # fn = score(xi, jnp.atleast_1d(ti))
            #     f, vjp_fn = jax.vjp(fn, xi)
            #     (eps_dfdy,) = vjp_fn(eps)
            #     logp = jnp.sum(eps_dfdy * eps)
            #     return f, logp

            def score_args(ti, xi, args):
                return score(xi, jnp.atleast_1d(ti))

            term = dfx.ODETerm(score_args)
            # solver = dfx.Heun()
            _, step_rng = jax.random.split(rng.key)
            # eps = random.normal(key, y.shape)
            # delta_log_likelihood = 0.0
            solver = dfx.Dopri5()
            sol = dfx.diffeqsolve(
                term, solver, t0, t1, dt0, x, saveat=SaveAt(t1=True, ts=ts)
            )
            return sol.ys

        yt = vmap(f)(initial_samples)
        return yt[:, -1, :], jnp.moveaxis(yt, 0, 1)

    @partial(jit, static_argnums=[0])
    def loss(self, params, batch, batch_prior, batch_stats, rng):
        """Loss function for training the CFM score."""
        sigma_noise = 1e-3
        rng, step_rng = random.split(rng)
        N_batch = batch.shape[0]

        t = random.uniform(step_rng, (N_batch, 1))
        x0 = batch_prior
        x1 = batch
        noise = random.normal(step_rng, (N_batch, self.ndims))
        psi_0 = t * batch + (1 - t) * batch_prior + sigma_noise * noise
        output, updates = self.state.apply_fn(
            {"params": params, "batch_stats": batch_stats},
            psi_0,
            t,
            train=True,
            mutable=["batch_stats"],
        )
        psi = x1 - x0
        loss = jnp.mean((output - psi) ** 2 + normal_log_likelihood(batch_prior))
        # dfx.ODETerm
        # term = dfx.ODETerm(approx_logp_wrapper)
        # solver = dfx.Dopri5()
        # with jax.disable_jit():
        #     sol = dfx.diffeqsolve(term, solver, 1.0,0.0, -1e-3, batch_prior, (noise,self.state.apply_fn))
        return loss, updates
