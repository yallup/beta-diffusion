from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import anesthetic as ns
import jax
import jax.numpy as jnp
import jax.random as random
import optax
from clax import ClassifierSamples
from flax import linen as nn
from flax import traverse_util
from flax.training.train_state import TrainState
from jax import jit, tree_map
from jaxopt import LBFGS
from optax import tree_utils as otu
from scipy.stats import multivariate_normal
from tqdm import tqdm

from fusions.network import Classifier, ScoreApprox
from fusions.optimal_transport import FullOT, NullOT, PriorExtendedNullOT

# from clax import ClassifierSamples
# from optax.contrib import reduce_on_plateau


@dataclass
class Trace:
    iteration: int = field(default=0)
    losses: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    calibrate_losses: list[float] = field(default_factory=list)


class Model(ABC):
    """
    Base class for models.
    """

    def __init__(self, prior=None, n=None, **kwargs) -> None:
        self.prior = prior
        self.rng = random.PRNGKey(kwargs.get("seed", 2023))
        self.noise = kwargs.pop("noise", 1e-3)
        if not self.prior:
            if not n:
                raise ValueError("Either prior or n must be specified.")
            self.rng, step_rng = random.split(self.rng)
            self.prior = multivariate_normal(jnp.zeros(n))

        # self.map = kwargs.get("map", NullOT)
        self.map = PriorExtendedNullOT
        # self.map = FullOT

        self.state = None
        self.calibrate_state = None
        self.trace = None

    @abstractmethod
    def reverse_process(self, initial_samples, score, rng, **kwargs):
        pass

    def sample_prior(self, n):
        """Sample from the prior distribution.

        Args:
            n (int): Number of samples to draw.

        Returns:
            jnp.ndarray: Samples from the prior distribution.
        """
        return self.prior.rvs(n).reshape(-1, self.ndims)

    def predict(self, initial_samples, **kwargs):
        """Run the diffusion model on user-provided samples.

        Args:
            initial_samples (jnp.ndarray): Samples to run the model on.

        Keyword Args:
            steps (int): Number of aditional time steps to save at.
            jac (bool): If True, return the jacobian of the process as well as the output (tuple).
            solution (str): Method to use for the jacobian. Defaults to "exact".
                        one of "exact", "none", "approx".

        Returns:
            jnp.ndarray: Samples from the posterior distribution.
        """
        jac = kwargs.pop("jac", False)
        steps = kwargs.pop("steps", 0)
        solution = kwargs.pop("solution", "none")
        rng = kwargs.pop("rng", self.rng)
        # self.rng, step_rng = random.split(self.rng)
        x, j = self.reverse_process(
            initial_samples,
            self._predict,
            rng=rng,
            steps=steps,
            solution=solution,
            **kwargs,
        )  # , step_rng)
        if jac:
            return x.squeeze(), j.squeeze()
        else:
            return x.squeeze()

    def sample_posterior(self, n, **kwargs):
        """Draw samples from the posterior distribution.

        Args:
            n (int): Number of samples to draw.

        Keyword Args:
            steps (int): Number of aditional time steps to save at.
            jac (bool): If True, return the jacobian of the process as well as the output (tuple).
            solution (str): Method to use for the jacobian. Defaults to "exact".
                        one of "exact", "none", "approx".


        Returns:
            jnp.ndarray: Samples from the posterior distribution.
        """
        self.rng, step_rng = random.split(self.rng)
        return self.predict(self.sample_prior(n), rng=step_rng, **kwargs)

    def score_model(self):
        """Score model for training the diffusion model."""
        return ScoreApprox()

    def classifier_model(self):
        """Score model for training the diffusion model."""
        return Classifier()

    def rvs(self, n, **kwargs):
        """Alias for sample_posterior.

        Args:
            n (int): Number of samples to draw.

        Keyword Args:
            see sample_posterior and predict.

        Returns:
            jnp.ndarray: Samples from the posterior distribution.
        """
        return self.sample_posterior(n, **kwargs)

    def _train(self, data, **kwargs):
        """Internal wrapping of training loop."""
        self.trace = Trace()
        # batch_size = kwargs.get("batch_size", 256)
        # n_epochs = kwargs.get("n_epochs", data.shape[0])
        prior_samples = kwargs.get("prior_samples", None)
        batch_size = kwargs.get("batch_size")
        batches_per_epoch = kwargs.pop("batches_per_epoch")
        epochs = kwargs.get("epochs", 10)

        @jit
        def update_step(state, batch, batch_prior, rng):
            val, grads = jax.value_and_grad(self.loss)(
                state.params, batch, batch_prior, rng
            )
            state = state.apply_gradients(grads=grads)  # , scale_value=val)
            # state = state.replace(batch_stats=updates["batch_stats"])
            # state = state.replace(value=val)
            return val, state

        train_size = data.shape[0]
        if prior_samples is None:
            prior_samples = jnp.array(
                self.prior.rvs(train_size).reshape(-1, self.ndims)
                # self.prior.rvs(train_size * 100).reshape(-1, self.ndims)
            )
        batch_size = min(batch_size, train_size)

        losses = []
        map = self.map(prior_samples, data)
        tepochs = tqdm(range(epochs))
        for k in tepochs:
            epoch_losses = []
            for _ in range(batches_per_epoch):
                self.rng, step_rng = random.split(self.rng)
                perm, perm_prior = map.sample(batch_size)
                batch = data[perm]
                batch_label = prior_samples[perm_prior]
                loss, self.state = update_step(self.state, batch, batch_label, step_rng)
                epoch_losses.append(loss)

            epoch_summary_loss = jnp.mean(jnp.asarray(epoch_losses))
            tepochs.set_postfix(loss="{:.2e}".format(epoch_summary_loss))
            losses.append(epoch_summary_loss)
            # if losses[::-1][:patience] < epoch_summary_loss:
            #     break
            self.trace.losses = jnp.asarray(losses)

    def _init_state(self, **kwargs):
        """Initialise the state of the training."""
        prev_params = kwargs.get("params", None)
        dummy_x = jnp.zeros((1, self.ndims))
        dummy_t = jnp.ones((1, 1))
        self.rng, step_rng = random.split(self.rng)
        _params = self.score_model().init(step_rng, dummy_x, dummy_t)
        optimizer = kwargs.get("optimizer", None)
        lr = kwargs.get("lr", 1e-3)

        params = _params["params"]
        # batch_stats = _params["batch_stats"]

        target_batches_per_epoch = kwargs.pop("target_batches_per_epoch")
        warmup_fraction = kwargs.get("warmup_fraction", 0.05)
        cold_fraction = kwargs.get("cold_fraction", 0.05)
        cold_lr = kwargs.get("cold_lr", 1e-3)
        epochs = kwargs.pop("epochs")

        self.schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=int(warmup_fraction * target_batches_per_epoch * epochs + 1),
            decay_steps=int(
                (1 - cold_fraction) * target_batches_per_epoch * epochs + 1
            ),
            end_value=lr * cold_lr,
            exponent=1.0,
        )
        if not optimizer:
            optimizer = optax.chain(
                # optax.adaptive_grad_clip(0.01),
                # optax.contrib.schedule_free_adamw(lr, warmup_steps=transition_steps)
                optax.adamw(self.schedule),
                # optax.adamw(lr),
            )

        if prev_params:
            _params = {}
            _params["params"] = prev_params
            # # _params["params"] = params
            # lr = 1e-3
            # # _params["batch_stats"] = stats
            # last_layer = list(_params["params"].keys())[-1]
            # optimizer = optax.chain(
            #     optax.clip_by_global_norm(1.0),
            #     optax.adamw(
            #         optax.cosine_decay_schedule(
            #             lr * 5, transition_steps * 10, alpha=lr * 1e-2
            #         )
            #     ),
            #     #     optax.contrib.reduce_on_plateau(
            #     #         factor=0.5,
            #     #         patience=transition_steps // 10,  # 10
            #     #         # cooldown=transition_steps // 10,  # 10
            #     #         accumulation_size=transition_steps,
            #     #     ),
            # )

        params = _params["params"]
        # batch_stats = _params["batch_stats"]
        self.state = TrainState.create(
            apply_fn=self.score_model().apply,
            params=params,
            # batch_stats=batch_stats,
            tx=optimizer,
            # losses=[],
            # val = 1e-1
        )

    @abstractmethod
    def loss(self, params, batch, batch_prior, batch_stats, rng):
        """Loss function for training the diffusion model."""
        pass

    def predict_weight(self, samples, **kwargs):
        prob = kwargs.pop("prob", False)
        if prob:
            return nn.softmax(self._predict_weight(samples, **kwargs))
        else:
            return self._predict_weight(samples, **kwargs)

    def train(self, data, **kwargs):
        """Train the diffusion model on the provided data.

        Args:
            data (jnp.ndarray): Data to train on.

        Keyword Args:
            restart (bool): If True, reinitialise the model before training. Defaults to False.
            batch_size (int): Size of the training batches. Defaults to 128.
            epochs (int): Number of training epochs. Defaults to 1000.
            lr (float): Learning rate. Defaults to 1e-3.
        """
        restart = kwargs.get("restart", False)
        self.noise = kwargs.get("noise", 1e-3)
        self.ndims = data.shape[-1]
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        batch_size = kwargs.get("batch_size", 128)
        data_size = data.shape[0]
        batch_size = min(batch_size, data_size)
        kwargs["batch_size"] = batch_size
        batches_per_epoch = data_size // batch_size
        kwargs["batches_per_epoch"] = batches_per_epoch
        # data = (data - self.mean) / self.std

        if not self.prior:
            self.prior = multivariate_normal(
                key=random.PRNGKey(0), mean=jnp.zeros(self.ndims)
            )
        # data = self.chains.sample(200).to_numpy()[..., :-3]
        if (not self.state) | restart:
            kwargs["target_batches_per_epoch"] = batches_per_epoch
            self._init_state(**kwargs)
        else:
            kwargs["target_batches_per_epoch"] = batches_per_epoch
            self._init_state(
                params=self.state.params,  # batch_stats=self.state.batch_stats
                **kwargs,
            )
        # self._init_state=self._init_state.replace(grads=jax.tree_map(jnp.zeros_like, self._init_state.params))
        # self.state.params.replace(grads=jax.tree_map(jnp.zeros_like, self.state.params))
        # self.state.replace(grads=jax.tree_map(jnp.zeros_like, self.state.params))
        # lr = kwargs.get("lr", 1e-3)
        # self.state.tx = optax.adam(lr)
        self._train(data, **kwargs)
        self._predict = lambda x, t: self.state.apply_fn(
            {
                "params": self.state.params,
                # "batch_stats": self.state.batch_stats,
            },
            x,
            t,
            train=False,
            # rngs={"dropout": self.rng},
        )

    def calibrate(self, samples_a, samples_b, **kwargs):
        """Calibrate the model on the provided data.

        Args:
            samples_a (jnp.ndarray): Samples to train on.
            samples_b (jnp.ndarray): Samples to train on.

        Keyword Args:
            restart (bool): If True, reinitialise the model before training. Defaults to False.
            batch_size (int): Size of the training batches. Defaults to 128.
            n_epochs (int): Number of training epochs. Defaults to 1000.
            lr (float): Learning rate. Defaults to 1e-3.
        """
        self.calibrator = ClassifierSamples()
        self.calibrator.train(samples_a, samples_b, **kwargs)
        self._predict_weight = lambda x: self.calibrator.predict(x)
