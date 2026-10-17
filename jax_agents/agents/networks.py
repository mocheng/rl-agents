from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


class MLP(nn.Module):
    """
        A multi-layer perceptron over flattened observations.
    """

    layers: Sequence[int] = (256, 256)
    num_actions: int = 4

    @nn.compact
    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        for size in self.layers:
            x = nn.relu(nn.Dense(size)(x))
        return nn.Dense(self.num_actions)(x)


class ConvEncoder(nn.Module):
    """
        A convolutional encoder for grid observations (batch, rows, cols, channels),
        e.g. Sokoban's (10, 10, 2) grid, followed by a small MLP head.
    """

    channels: Sequence[int] = (32, 64)
    head_layers: Sequence[int] = (128,)
    num_actions: int = 4

    @nn.compact
    def __call__(self, x):
        for channels in self.channels:
            x = nn.relu(nn.Conv(features=channels, kernel_size=(3, 3), padding="SAME")(x))
        x = x.reshape(x.shape[0], -1)
        for size in self.head_layers:
            x = nn.relu(nn.Dense(size)(x))
        return nn.Dense(self.num_actions)(x)


def network_factory(config, num_actions):
    """
        Build a value network from a model configuration, following the repository's
        model configuration convention: {"type": "MLP", "layers": [...]} or
        {"type": "ConvEncoder", "channels": [...], "head_layers": [...]}.
    """
    model_type = config.get("type", "MLP")
    if model_type == "MLP":
        return MLP(layers=config.get("layers", [256, 256]), num_actions=num_actions)
    if model_type == "ConvEncoder":
        return ConvEncoder(channels=config.get("channels", [32, 64]),
                           head_layers=config.get("head_layers", [128]),
                           num_actions=num_actions)
    raise ValueError("Unknown JAX model type: {}".format(model_type))
