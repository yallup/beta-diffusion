from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, jit, random, vmap
from jax.lax import scan

from fusions.model import Model


class Diffusion(Model):
    beta_min: float = 1e-3
    beta_max: float = 3.0
    steps: int = 1000
    train_ts = jnp.arange(1, steps) / (steps - 1)
    # train_ts=jnp.geomspace(beta_min,beta_max,steps)

    def beta_t(self, t):
        """Beta function of the diffusion model."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def alpha_t(self, t):
        """Alpha function of the diffusion model."""
        return t * self.beta_min + 0.5 * t**2 * (self.beta_max - self.beta_min)

    def sample_ts(self, n):
        return n

    def mean_factor(self, t):
        """Mean factor of the diffusion model."""
        return jnp.exp(-0.5 * self.alpha_t(t))

    def var(self, t):
        """Variance of the diffusion model."""
        return 1 - jnp.exp(-self.alpha_t(t))

    def drift(self, x, t):
        """Drift of the diffusion model."""
        return -0.5 * self.beta_t(t) * x

    def dispersion(self, t):
        """Dispersion of the diffusion model."""
        return jnp.sqrt(self.beta_t(t))

    @partial(jit, static_argnums=[0, 2])
    def reverse_process(self, initial_samples, score, rng):
        """Run the reverse SDE.

        Args:
            initial_samples (jnp.ndarray): Samples to run the model on.
            score (callable): Score function.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Samples from the posterior distribution. and the history of the process.
        """

        def f(carry, params):
            t, dt = params
            x, rng = carry
            rng, step_rng = jax.random.split(rng)
            disp = self.dispersion(1 - t)
            t = jnp.ones((x.shape[0], 1)) * t
            drift = -self.drift(x, 1 - t) + disp**2 * score(x, 1 - t)
            step_rng, noise_rng = jax.random.split(step_rng)
            noise = random.normal(noise_rng, x.shape)
            x = x + dt * drift + jnp.sqrt(dt) * disp * noise
            return (x, rng), (carry)

        rng, step_rng = random.split(rng)
        # initial_samples = random.normal(rng, initial_samples.shape)
        rng, step_rng = random.split(step_rng)
        dts = self.train_ts[1:] - self.train_ts[:-1]
        params = jnp.stack([self.train_ts[:-1], dts], axis=1)
        (x, _), (x_t, _) = scan(f, (initial_samples, step_rng), params)
        return x, x_t

    @partial(jit, static_argnums=[0])
    def loss(self, params, batch, batch_prior, batch_stats, rng):
        """Loss function for training the diffusion model."""
        rng, step_rng = random.split(rng)
        N_batch = batch.shape[0]
        t = random.randint(step_rng, (N_batch, 1), 1, self.steps) / (self.steps - 1)
        # alpha = 2.0
        # t = 1 - (t) ** (1 / alpha)
        mean_coeff = self.mean_factor(t)
        vs = self.var(t)
        stds = jnp.sqrt(vs)
        rng, step_rng = random.split(rng)
        noise = batch_prior
        # noise = batch_prior + random.normal(step_rng, batch.shape)
        # noise = random.normal(step_rng, batch.shape)
        xt = batch * mean_coeff + noise * stds
        output, updates = self.state.apply_fn(
            {"params": params, "batch_stats": batch_stats},
            xt,
            t,
            train=True,
            mutable=["batch_stats"],
        )

        loss = jnp.mean((noise + output * stds) ** 2)
        return loss, updates
