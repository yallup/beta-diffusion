from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
from flax.training import train_state

# class DataLoader(object):
#     def __init__(self, data, batch_size, rng, shuffle=True) -> None:
#         self.data = jnp.array(data)
#         self.rng = rng
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.i = 0

#     def __iter__(self):
#         return self

#     def __next__(self):
#         if self.i >= len(self.data):
#             self.i = 0
#             if self.shuffle:
#                 self.rng, rng = jax.random.split(self.rng)
#                 perm = jax.random.permutation(rng, len(self.data))
#                 self.data = self.data[perm]
#         batch = self.data[self.i : self.i + self.batch_size]
#         self.i += self.batch_size
#         if len(batch) != self.batch_size:
#             raise StopIteration
#         return batch


class TrainState(train_state.TrainState):
    batch_stats: Any
    losses: Any


def zeros_init(key, shape, dtype=jnp.float32):
    return jnp.zeros(shape, dtype)


class Classifier(nn.Module):
    """A simple MLP classifier."""

    n_initial: int = 256
    n_hidden: int = 32
    n_layers: int = 3
    act = nn.leaky_relu

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Dense(self.n_initial)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.silu(x)
        for i in range(self.n_layers):
            x = nn.Dense(self.n_hidden)(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.silu(x)
        x = nn.Dense(1)(x)
        return x


class ScoreApprox(nn.Module):
    """A simple model with multiple fully connected layers and some fourier features for the time variable."""

    n_initial: int = 256
    n_hidden: int = 32
    n_layers: int = 3
    act = nn.leaky_relu

    @nn.compact
    def __call__(self, x, t, train: bool):
        in_size = x.shape[-1]
        # act = nn.relu
        # # t = jnp.concatenate([t - 0.5, jnp.cos(2 * jnp.pi * t)], axis=1)
        t = jnp.concatenate(
            [
                t - 0.5,
                jnp.cos(2 * jnp.pi * t),
                jnp.sin(2 * jnp.pi * t),
                -jnp.cos(4 * jnp.pi * t),
            ],
            axis=-1,
        )
        x = jnp.concatenate([x, t], axis=-1)
        x = nn.Dense(self.n_initial)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.silu(x)
        for i in range(self.n_layers):
            x = nn.Dense(self.n_hidden)(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.silu(x)
        x = nn.Dense(in_size, kernel_init=zeros_init)(x)
        return x


class unetConv(nn.Module):
    out_dim: int

    # Conv Parameters
    kernel_size: tuple = (3, 3)
    strides: int = 1
    padding: int = 0

    use_batchnorm: bool = False

    # BatchNorm Parameters
    use_running_average: bool = False
    momentum: float = 0.9
    epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        if self.use_batchnorm:
            x = nn.Conv(
                features=self.out_dim,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
            )(x)
            x = nn.BatchNorm(
                use_running_average=self.use_running_average,
                momentum=self.momentum,
                epsilon=self.epsilon,
                dtype=self.dtype,
            )(x)
            x = nn.relu(x)
            x = nn.Conv(
                features=self.out_size,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
            )(x)
            x = nn.BatchNorm(
                use_running_average=self.use_running_average,
                momentum=self.momentum,
                epsilon=self.epsilon,
                dtype=self.dtype,
            )(x)
            x = nn.relu(x)
            return x
        else:
            x = nn.Conv(
                features=self.out_dim,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
            )(x)
            x = nn.relu(x)
            x = nn.Conv(
                features=self.out_dim,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
            )(x)
            x = nn.relu(x)
            return x


class Upsample(nn.Module):
    out_dim: int
    is_deconv: bool

    # ConvTranspose Parameters
    kernel_size: int = 2
    strides: int = 2

    @nn.compact
    def forward(self, inputs1, inputs2):
        if self.is_deconv:
            outputs2 = nn.ConvTranspose(
                features=self.out_dim,
                kernel_size=self.kernel_size,
                strides=self.strides,
            )(inputs2)
        else:
            outputs2 = nn.UpsamplingBilinear2d(scale_factor=2)(inputs2)

        offset = outputs2.size()[2] - inputs1.size()[2]

        padding = 2 * [offset // 2, offset // 2]

        outputs1 = jnp.pad(inputs1, padding)

        return unetConv(features=self.out_dim, is_batchnorm=False)(
            jnp.concatenate([outputs1, outputs2], 1)
        )


class unet(nn.Module):
    feature_scale: int = 4
    n_classes: int = 21
    is_deconv: bool = True
    use_batchnorm: bool = True
    kernel_size: int = 2

    @nn.compact
    def __call__(self, x):
        is_deconv = self.is_deconv
        use_batchnorm = self.use_batchnorm
        feature_scale = self.feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]

        # downsampling

        conv1 = unetConv(filters[0], use_batchnorm)(x)
        maxpool1 = nn.MaxPool2d(kernel_size=self.kernel_size)(conv1)

        conv2 = unetConv(filters[1], use_batchnorm)(maxpool1)
        maxpool2 = nn.MaxPool2d(kernel_size=self.kernel_size)(conv2)

        conv3 = unetConv(filters[2], use_batchnorm)(maxpool2)
        maxpool3 = nn.MaxPool2d(kernel_size=self.kernel_size)(conv3)

        conv4 = unetConv(filters[3], use_batchnorm)(maxpool3)
        maxpool4 = nn.MaxPool2d(kernel_size=self.kernel_size)(conv4)

        center = unetConv(filters[4], use_batchnorm)(maxpool4)

        # upsampling
        up4 = Upsample(filters[3], is_deconv=is_deconv)(conv4, center)
        up3 = Upsample(filters[2], is_deconv=is_deconv)(conv3, up4)
        up2 = Upsample(filters[1], is_deconv=is_deconv)(conv2, up3)
        up1 = Upsample(filters[0], is_deconv=is_deconv)(conv1, up2)

        # final conv (without any concat)
        final = nn.Conv(self.n_classes, 1)(up1)

        return final
