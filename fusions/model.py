from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import anesthetic as ns
import jax
import jax.numpy as jnp
import jax.random as random
import optax
from flax import linen as nn
from flax import traverse_util
from jax import jit, tree_map
from optax import tree_utils as otu
from scipy.stats import multivariate_normal
from tqdm import tqdm

from fusions.network import Classifier, ScoreApprox, TrainState
from fusions.optimal_transport import NullOT, PriorExtendedNullOT

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
        batch_size = kwargs.get("batch_size", 256)
        n_epochs = kwargs.get("n_epochs", data.shape[0])
        prior_samples = kwargs.get("prior_samples", None)

        @jit
        def update_step(state, batch, batch_prior, rng):
            (val, updates), grads = jax.value_and_grad(self.loss, has_aux=True)(
                state.params, batch, batch_prior, rng
            )
            state = state.apply_gradients(grads=grads, value=val)
            # state = state.replace(batch_stats=updates["batch_stats"])
            # state = state.replace(value=val)
            return val, state

        train_size = data.shape[0]
        if prior_samples is None:
            prior_samples = jnp.array(
                self.prior.rvs(train_size * 100).reshape(-1, self.ndims)
            )
        batch_size = min(batch_size, train_size)

        losses = []
        map = self.map(prior_samples, data)
        tepochs = tqdm(range(n_epochs))
        for k in tepochs:
            self.rng, step_rng = random.split(self.rng)
            perm_prior, perm = map.sample(batch_size)
            batch = data[perm, :]
            batch_prior = prior_samples[perm_prior, :]
            loss, self.state = update_step(self.state, batch, batch_prior, step_rng)
            # self.trace.losses.append(loss)
            losses.append(loss)

            # losses.append(loss)
            if (k + 1) % 10 == 0:
                ma = jnp.mean(jnp.array(losses[-10:]))
                self.trace.losses.append(ma)
                tepochs.set_postfix(loss=ma)
                self.trace.iteration += 1
                lr_scale = otu.tree_get(self.state, "scale")
                self.trace.lr.append(lr_scale)
                # if lr_scale < 1e-1:
                #     break

    def _train_calibrator(self, data_a, data_b, **kwargs):
        """Internal wrapping of training loop."""
        batch_size = kwargs.get("batch_size", 512)
        n_epochs = kwargs.get("n_epochs", 50)

        @jit
        def update_step(state, batch, batch_labels, rng):
            val, grads = jax.value_and_grad(self.calibrate_loss)(
                state.params, batch, batch_labels
            )
            state = state.apply_gradients(grads=grads)
            # state = state.replace(batch_stats=updates["batch_stats"])
            return val, state

        train_size = data_a.shape[0]

        # if self.prior:
        #     prior_samples = jnp.array(self.prior.rvs(train_size))
        # else:
        #     prior_samples = jnp.zeros_like(data)

        batch_size = min(batch_size, train_size)
        n_batches = train_size // batch_size
        labels_a = jnp.zeros(data_a.shape[0])
        labels_b = jnp.ones(data_b.shape[0])
        labels = jnp.concatenate([labels_a, labels_b])
        labels = jnp.asarray(labels, dtype=int)
        data = jnp.concatenate([data_a, data_b])

        losses = []
        tepochs = tqdm(range(n_epochs))
        for k in tepochs:
            self.rng, step_rng = random.split(self.rng)
            perm = random.permutation(step_rng, jnp.arange(data.shape[0]))
            data = data[perm, :]
            labels = labels[perm]

            for i in range(n_batches):
                self.rng, step_rng = random.split(self.rng)
                # Get the indices of the current batch
                start = i * batch_size
                end = min(start + batch_size, train_size)

                # Extract the batch from the shuffled data and labels
                batch = data[start:end, :]
                batch_labels = labels[start:end]

                loss, self.calibrate_state = update_step(
                    self.calibrate_state, batch, batch_labels, step_rng
                )
                losses.append(loss)

                # losses.append(loss)
                if (k + 1) % 10 == 0:
                    ma = jnp.mean(jnp.array(losses[-10:]))
                    self.trace.calibrate_losses.append(ma)
                    tepochs.set_postfix(loss=ma)

    def _init_state(self, **kwargs):
        """Initialise the state of the training."""
        prev_params = kwargs.get("params", None)
        dummy_x = jnp.zeros((1, self.ndims))
        dummy_t = jnp.ones((1, 1))
        self.rng, step_rng = random.split(self.rng)
        _params = self.score_model().init(step_rng, dummy_x, dummy_t)
        params = _params["params"]

        lr = kwargs.get("lr", 1e-2)
        base_learning_rate = lr

        transition_steps = kwargs.get("transition_steps", 100)

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                optax.cosine_decay_schedule(lr, transition_steps * 10, alpha=lr * 1e-2)
            ),
            # optax.contrib.reduce_on_plateau(
            #     factor=0.5,
            #     patience=transition_steps // 10,  # 10
            #     # cooldown=transition_steps // 10,  # 10
            #     accumulation_size=transition_steps,
            # ),
        )

        if prev_params:
            _params = {}
            _params["params"] = prev_params
            # _params["params"] = params
            lr = 1e-3
            # _params["batch_stats"] = stats
            last_layer = list(_params["params"].keys())[-1]
            optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(
                    optax.cosine_decay_schedule(
                        lr * 5, transition_steps * 10, alpha=lr * 1e-2
                    )
                ),
                #     optax.contrib.reduce_on_plateau(
                #         factor=0.5,
                #         patience=transition_steps // 10,  # 10
                #         # cooldown=transition_steps // 10,  # 10
                #         accumulation_size=transition_steps,
                #     ),
            )

        params = _params["params"]
        self.state = TrainState.create(
            apply_fn=self.score_model().apply,
            params=params,
            # batch_stats=batch_stats,
            tx=optimizer,
            losses=[],
            # val = 1e-1
        )

    def _init_calibrate_state(self, **kwargs):
        dummy_x = jnp.zeros((1, self.ndims))

        _params = self.classifier_model().init(self.rng, dummy_x)
        lr = kwargs.get("lr", 1e-2)
        optimizer = optax.adam(lr)
        params = _params["params"]
        # batch_stats = _params["batch_stats"]
        transition_steps = kwargs.get("transition_steps", 100)
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                optax.cosine_decay_schedule(lr * 5, transition_steps * 2, alpha=lr)
            ),
            optax.contrib.reduce_on_plateau(
                factor=0.5,
                patience=transition_steps // 10,  # 10
                # cooldown=transition_steps // 10,  # 10
                accumulation_size=transition_steps,
            ),
        )
        optimizer = optax.chain(optax.adamw(optax.cosine_decay_schedule(lr, 1000)))
        self.calibrate_state = TrainState.create(
            apply_fn=self.classifier_model().apply,
            params=params,
            # batch_stats=batch_stats,
            tx=optimizer,
            losses=[],
        )

    @abstractmethod
    def loss(self, params, batch, batch_prior, batch_stats, rng):
        """Loss function for training the diffusion model."""
        pass

    def calibrate_loss(self, params, batch, labels):
        """Loss function for training the calibrator."""
        output = self.calibrate_state.apply_fn(
            {"params": params},
            batch,
            # train=True,
            # mutable=["batch_stats"],
        )

        loss = optax.softmax_cross_entropy_with_integer_labels(
            output.squeeze(), labels
        ).mean()
        return loss

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
            n_epochs (int): Number of training epochs. Defaults to 1000.
            lr (float): Learning rate. Defaults to 1e-3.
        """
        restart = kwargs.get("restart", False)
        self.noise = kwargs.get("noise", 1e-3)
        self.ndims = data.shape[-1]
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)

        # data = (data - self.mean) / self.std

        if not self.prior:
            self.prior = multivariate_normal(
                key=random.PRNGKey(0), mean=jnp.zeros(self.ndims)
            )
        # data = self.chains.sample(200).to_numpy()[..., :-3]
        if (not self.state) | restart:
            self._init_state(**kwargs)
        else:
            self._init_state(
                params=self.state.params  # batch_stats=self.state.batch_stats
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
            # train=False,
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
        restart = kwargs.get("restart", False)
        self.ndims = samples_a.shape[-1]
        # data = self.chains.sample(200).to_numpy()[..., :-3]
        if (not self.calibrate_state) | restart:
            self._init_calibrate_state(**kwargs)
        # self._init_state=self._init_state.replace(grads=jax.tree_map(jnp.zeros_like, self._init_state.params))
        # self.state.params.replace(grads=jax.tree_map(jnp.zeros_like, self.state.params))
        # self.state.replace(grads=jax.tree_map(jnp.zeros_like, self.state.params))
        self._train_calibrator(samples_a, samples_b, **kwargs)
        self._predict_weight = lambda x: self.calibrate_state.apply_fn(
            {
                "params": self.calibrate_state.params,
                # "batch_stats": self.calibrate_state.batch_stats,
            },
            x,
            # train=False,
        )
