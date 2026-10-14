import logging
from pathlib import Path

import numpy as np
from gymnasium import spaces

from rl_agents.agents.common.abstract import AbstractAgent

logger = logging.getLogger(__name__)


def algorithm_factory(name):
    """
        Get a Stable-Baselines3 algorithm class by name.

        Custom algorithms defined in `rl_agents.agents.sb3.algorithms` are also supported.

    :param name: the algorithm name, e.g. DQN, PPO or MSEDQN
    :return: the algorithm class
    """
    import stable_baselines3
    if hasattr(stable_baselines3, name):
        return getattr(stable_baselines3, name)
    from rl_agents.agents.sb3 import algorithms
    if hasattr(algorithms, name):
        return getattr(algorithms, name)
    raise ValueError("Unknown Stable-Baselines3 algorithm: {}".format(name))


def make_model(config, env, seed=None, tensorboard_log=None):
    """
        Build a Stable-Baselines3 model from a configuration, on a given environment or vectorized
        environment.

    :param config: agent configuration with 'algo', 'policy' and optional 'model_kwargs', 'device'
    :param env: an environment or vectorized environment
    :param seed: optional random seed
    :param tensorboard_log: optional TensorBoard log directory
    :return: the Stable-Baselines3 model
    """
    algorithm = algorithm_factory(config["algo"])
    kwargs = dict(config.get("model_kwargs", {}))
    if "policy_kwargs" in config:
        kwargs["policy_kwargs"] = config["policy_kwargs"]
    if seed is not None:
        kwargs["seed"] = seed
    if tensorboard_log is not None:
        kwargs["tensorboard_log"] = tensorboard_log
    return algorithm(config.get("policy", "MlpPolicy"), env,
                     device=config.get("device", "auto"), **kwargs)


class SB3Agent(AbstractAgent):
    """
        A Stable-Baselines3 model exposed through this repository's agent interface.

        Training is performed by Stable-Baselines3 itself (`scripts/sb3_train.py` and
        `model.learn`). This agent is used for evaluation with `Evaluation`: it loads a saved model
        and acts greedily in test mode.
    """

    def __init__(self, env, config=None):
        super().__init__(config)
        self.env = env
        self.model = None
        self.training = True
        self.seed_value = None

    @classmethod
    def default_config(cls):
        return dict(algo="DQN",
                    policy="MlpPolicy",
                    model_kwargs=dict(),
                    device="auto")

    def act(self, state):
        if self.model is None:
            raise RuntimeError("No Stable-Baselines3 model loaded. Use --recover-from <model.zip>.")
        action, _ = self.model.predict(np.asarray(state), deterministic=not self.training)
        if isinstance(self.model.action_space, spaces.Discrete):
            return int(np.asarray(action).item())
        return np.asarray(action, dtype=np.float64)

    def plan(self, state):
        return [self.act(state)]

    def record(self, state, action, reward, next_state, done, info):
        # Training is handled by the Stable-Baselines3 training loop, not by the trainer
        pass

    def reset(self):
        pass

    def seed(self, seed=None):
        self.seed_value = seed
        if self.model is not None:
            self.model.set_random_seed(seed)
        return [seed]

    def eval(self):
        self.training = False

    def save(self, filename):
        path = self._zip_path(filename)
        self.model.save(path)
        return path

    def load(self, filename):
        path = self._zip_path(filename)
        self.model = algorithm_factory(self.config["algo"]).load(
            path, env=self.env, device=self.config.get("device", "auto"))
        self.model.set_random_seed(self.seed_value)
        logger.info("Loaded Stable-Baselines3 {} model from {}".format(self.config["algo"], path))
        return path

    @staticmethod
    def _zip_path(filename):
        path = Path(filename)
        if path.suffix != ".zip":
            path = path.with_suffix(".zip")
        return path
