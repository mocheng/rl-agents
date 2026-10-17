import logging
import os
import pickle

import jax
import numpy as np

from rl_agents.agents.common.abstract import AbstractAgent

logger = logging.getLogger(__name__)

# Only the first JAX agent of a process sets the JAX default device, so that a process
# hosting several agents (e.g. tests) keeps a consistent placement policy.
_DEFAULT_DEVICE_IS_SET = False


def resolve_device(name="auto"):
    """
        Resolve a device configuration entry to a JAX device.

    :param name: "auto" (first available device, GPU when present), "cpu", or "gpu"
                 (also matches "cuda"/"metal" platforms).
    :return: a jax.Device
    """
    devices = jax.devices()
    if name == "auto":
        return devices[0]
    if name == "cpu":
        for device in devices:
            if device.platform == "cpu":
                return device
        return devices[0]
    if name in ("gpu", "cuda", "metal"):
        for device in devices:
            if device.platform in ("gpu", "cuda", "metal"):
                return device
        logger.warning("No GPU device found (requested '{}'). Falling back to {}".format(name, devices[0]))
        return devices[0]
    raise ValueError("Unknown device '{}'. Choose from: auto, cpu, gpu".format(name))


def to_npz_path(filename):
    """
        Normalize a checkpoint path to a .npz path, mirroring the repository convention
        that models are saved under `saved_models/<name>.tar`.
    """
    filename = str(filename)
    if filename.endswith(".tar"):
        return filename[:-len(".tar")] + ".npz"
    if not filename.endswith(".npz"):
        return filename + ".npz"
    return filename


class JaxAgent(AbstractAgent):
    """
        Base class of JAX-native agents.

        Implements the repository's agent interface (`act`, `record`, `seed`, `reset`,
        `save`, `load`, `eval`) on top of JAX pytrees. Training is performed by the agent
        itself through `train_jax(evaluation)` (called by `jax_agents.trainer.JaxEvaluation`),
        using jitted and vectorized rollouts. The single-step `act` allows the repository's
        usual evaluation loop (`Evaluation.run_episodes`, `experiments.py --test`) to run
        unchanged.

        Checkpoints are npz files: pytree leaves are stored as numpy arrays, and the pytree
        structure is pickled alongside, so any flax/optax pytree can be saved and restored.
    """

    batched = True

    def __init__(self, env, config=None):
        super().__init__(config)
        self.env = env
        self.device = resolve_device(self.config.get("device", "auto"))
        global _DEFAULT_DEVICE_IS_SET
        if not _DEFAULT_DEVICE_IS_SET:
            jax.config.update("jax_default_device", self.device)
            _DEFAULT_DEVICE_IS_SET = True
        self.key = jax.random.PRNGKey(0)
        self.training = True
        self._initialized = False

    @classmethod
    def default_config(cls):
        return dict(device="auto")

    def seed(self, seed=None):
        if seed is not None:
            self.key = jax.random.PRNGKey(seed)
        return [seed]

    def eval(self):
        self.training = False

    def record(self, state, action, reward, next_state, done, info):
        # Training is handled by the agent's own vectorized loop, not by per-step recording.
        pass

    def reset(self):
        pass

    def save(self, filename):
        """
            Save the agent state (a pytree) and extra scalars to a npz file.
        """
        filename = to_npz_path(filename)
        state, extras = self._get_state()
        leaves, treedef = jax.tree_util.tree_flatten(state)
        data = {
            "treedef": np.frombuffer(pickle.dumps(treedef), dtype=np.uint8),
            "num_leaves": np.asarray(len(leaves)),
        }
        for i, leaf in enumerate(leaves):
            data["leaf_{}".format(i)] = np.asarray(leaf)
        for name, value in (extras or {}).items():
            data[name] = np.asarray(value)
        np.savez(filename, **data)
        return filename

    def load(self, filename):
        """
            Load the agent state from a npz file.
        """
        filename = to_npz_path(filename)
        if not os.path.exists(filename):
            raise FileNotFoundError("No checkpoint found at {}".format(filename))
        with np.load(filename, allow_pickle=False) as data:
            treedef = pickle.loads(bytes(data["treedef"]))
            num_leaves = int(data["num_leaves"])
            leaves = [data["leaf_{}".format(i)] for i in range(num_leaves)]
            state = jax.tree_util.tree_unflatten(treedef, leaves)
            extras = {name: data[name].item() if data[name].ndim == 0 else data[name]
                      for name in data.files
                      if not name.startswith("leaf_") and name not in ("treedef", "num_leaves")}
        state = jax.tree_util.tree_map(lambda leaf: jax.device_put(leaf, self.device), state)
        self._set_state(state, extras)
        self._initialized = True
        return filename

    def _get_state(self):
        """
        :return: (state, extras) where `state` is a pytree of jax arrays and `extras` a
                 dict of scalar values, both restored by `_set_state`.
        """
        raise NotImplementedError

    def _set_state(self, state, extras):
        raise NotImplementedError

    def train_jax(self, evaluation):
        """
            Train the agent with jitted/vectorized rollouts, using the evaluation object
            for global step accounting, TensorBoard summaries and checkpointing.
        """
        raise NotImplementedError
