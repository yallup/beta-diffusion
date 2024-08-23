import math
from functools import partial

import diffrax as dfx
import jax.numpy as jnp
import jax.random as random
from diffrax import SaveAt
from jax import disable_jit, grad, jit, pmap, vjp, vmap

from fusions.model import Model

# from ott.geometry import pointcloud
# from ott.solvers import linear


class CFM(Model):
    """Continuous Flow Matching."""

    @partial(jit, static_argnums=[0, 2, 4, 5])
    def reverse_process(
        self,
        initial_samples,
        score,
        rng,
        steps=0,
        solution="none",
        t0=0.0,
        t=1.0,
        dt0=1e-3,
    ):
        """Run the reverse ODE.

        Args:
            initial_samples (jnp.ndarray): Samples to run the model on.
            score (callable): Score function.
            rng: Jax Random number generator key.

        Keyword Args:
            steps (int, optional) : Number of time steps to save in addition to t=1. Defaults to 0.
            solution (str, optional): Method to use for the jacobian. Defaults to "exact".
                        one of "exact", "none", "approx".

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Samples from the posterior distribution. and the history of the process.
        """
        ts = jnp.linspace(t0, t, steps)

        def solver_none(ti, conditions, args):
            xi, null_jac = conditions
            return score(xi, jnp.atleast_1d(ti)), null_jac

        def solver_exact(ti, conditions, args):
            xi, initial = conditions
            # overload_score = lambda x: score(x, jnp.atleast_1d(ti))
            f, vjp_f = vjp(score, xi, jnp.atleast_1d(ti))
            # f, vjp_f = vjp(overload_score, xi)
            size = xi.shape[0]
            eye = jnp.eye(size)
            (dfdx, _) = vmap(vjp_f)(eye)
            logp = jnp.trace(dfdx)
            return f, logp

        def solver_approx(ti, conditions, args):
            eps = args
            xi, _ = conditions
            eps = random.normal(rng, xi.shape)
            f, vjp_f = vjp(score, xi, jnp.atleast_1d(ti))
            eps_dfdx, _ = vjp_f(eps)
            logp = jnp.sum(eps_dfdx * eps)
            return f, logp

        def f(x, eps):
            # jacobian = jnp.zeros(x.shape[0])
            jacobian = 0.0
            conditions = (x, jacobian)
            if solution == "exact":
                term = dfx.ODETerm(solver_exact)
            elif solution == "approx":
                term = dfx.ODETerm(solver_approx)
            else:
                term = dfx.ODETerm(solver_none)
            # term = dfx.ODETerm(score_args_exact)
            # solver = dfx.Heun()
            solver = dfx.Dopri5()
            # solver = dfx.Dopri8()
            # solver = dfx.Tsit5()

            with disable_jit():
                sol = dfx.diffeqsolve(
                    term,
                    solver,
                    t0,
                    t,
                    dt0,
                    conditions,
                    args=eps,
                    saveat=SaveAt(t1=True, ts=ts),
                    stepsize_controller=dfx.PIDController(1e-4, 1e-4),
                )
            return sol.ys

        # batch_rngs = random.split(rng, initial_samples.shape[0])
        eps = random.normal(rng, initial_samples.shape)
        # f(initial_samples[0], eps[0])
        # solver_exact(0.0, (initial_samples[0], initial_samples[0]), eps[0])
        yt, jt = vmap(f)(initial_samples, eps)
        return yt.squeeze(), jt.squeeze()

    @partial(jit, static_argnums=[0])
    def loss(self, params, batch, batch_prior, rng):
        """Loss function for training the CFM score.

        Args:
            params (jnp.ndarray): Parameters of the model.
            batch (jnp.ndarray): Target batch.
            batch_prior (jnp.ndarray): Prior batch.
            batch_stats (Any): Batch statistics (batchnorm running totals).
            rng: Jax Random number generator key.

        """
        # sigma_noise = 1e-3
        rng, step_rng = random.split(rng)
        N_batch = batch.shape[0]

        # geom = pointcloud.PointCloud(batch, batch_prior)
        # M = jnp.asarray(linear.solve(geom).matrix.flatten())
        # M = M / M.sum()
        # idx = idx = jax_choice_without_replacement(rng, M.shape[0], N_batch, M)
        # idx = jnp.divmod(idx, N_batch)
        # batch = batch[idx[0]]
        # batch_prior = batch_prior[idx[1]]
        t = random.uniform(step_rng, (N_batch, 1))
        t = jnp.power(t, 2.0 / 3.0)
        x0 = batch_prior
        x1 = batch
        noise = random.normal(step_rng, (N_batch, self.ndims))
        # psi_0 = t * batch + (1 - t) * batch_prior + sigma_noise * noise
        psi_0 = t * batch + (1 - t) * batch_prior + self.noise * noise
        # psi_0 = t * batch + (1 - t) * batch_prior

        output = self.state.apply_fn(
            {"params": params},
            psi_0,
            t,
            # mutable=["batch_stats"],
            # rngs={'dropout': rng}
        )
        psi = x1 - x0
        loss = jnp.mean((output - psi) ** 2)
        return loss


import jax
import jax.numpy as jnp


def jax_choice_without_replacement(key, a, size, p):
    """
    JAX implementation of random choice without replacement for large dimensions.

    Parameters:
    - key: A JAX PRNGKey
    - a: The number of items to choose from (equivalent to M.shape[0])
    - size: The number of items to choose (equivalent to N_batch)
    - p: Probability array (equivalent to M)

    Returns:
    - An array of indices chosen without replacement
    """
    # Generate Gumbel noise
    z = -jnp.log(-jnp.log(jax.random.uniform(key, shape=(a,))))

    # Combine with log probabilities
    logp = jnp.log(p)

    # Use top_k to get the indices of the k largest elements
    _, indices = jax.lax.top_k(logp + z, size)

    return jnp.sort(indices)
