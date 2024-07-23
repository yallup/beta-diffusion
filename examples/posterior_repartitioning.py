import anesthetic as ns
import blackjax
import distrax
import jax
import jax.numpy as jnp
from jax import random

from fusions.cfm import CFM

rng = random.PRNGKey(0)
import os

os.makedirs("plots", exist_ok=True)
d = 2

# def likelihood(theta):
#     return -jnp.log(
#         1
#         + (1 - theta[..., 0]) ** 2
#         + 100 * (theta[..., 1] - theta[..., 0] ** 2) ** 2
#     )


# prior_bij_fwd = lambda u: 10 * u - 5
# prior_bij_rev = lambda x: (x + 5) / 10

# d = 2

# prior = distrax.Uniform(jnp.ones(d) * -5, jnp.ones(d) * 5)


prior = distrax.Normal(jnp.zeros(d), jnp.ones(d))
rng, theta_key = jax.random.split(rng)
true_theta = prior.sample(seed=theta_key) * 5


def target(theta):
    return -jnp.sum((theta - true_theta) ** 2 / 0.05**2)


adapt = blackjax.window_adaptation(
    blackjax.nuts,
    target,
    target_acceptance_rate=0.8,
    progress_bar=True,  # , initial_step_size=1e-2
)

start_theta = prior.sample(seed=rng)
rng, warmup_key = jax.random.split(rng)

### NUTS params:
num_warmup = 5000
num_samples = 3000

(last_state, parameters), _ = adapt.run(warmup_key, start_theta, num_warmup)
kernel = blackjax.nuts(target, **parameters).step


# HMC loop
def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return states, (
        infos.acceptance_rate,
        infos.is_divergent,
        infos.num_integration_steps,
    )


rng, sample_key = jax.random.split(rng)
states, infos = inference_loop(sample_key, kernel, last_state, num_samples)


class uniform(object):
    def __init__(self, rng):
        self.rng = rng

    def rvs(self, size):
        self.rng, step_rng = jax.random.split(self.rng)
        return prior._sample_n(step_rng, size)


mapping = CFM(uniform(rng))
mapping.train(states.position)

n_steps = 100
x, j = mapping.sample_posterior(2000, jac=True, solution="exact", steps=n_steps)

time_steps_to_plot = [0, 50, 75, 95, 100]
f, a = ns.make_2d_axes([0, 1], figsize=(10, 6))
for i in time_steps_to_plot:
    ns.MCMCSamples(x[:, i, :]).plot_2d(a, label=f"t={i/n_steps:.2f}")

# ns.MCMCSamples(x[:, -1, :]).plot_2d(a)
ns.MCMCSamples(states.position).plot_2d(a, label="HMC")

a.iloc[-1, 0].legend(
    loc="lower center",
    bbox_to_anchor=(len(a) / 2, len(a)),
    ncol=len(time_steps_to_plot) + 1,
)
f.savefig("plots/transport.pdf")
print("Done")
