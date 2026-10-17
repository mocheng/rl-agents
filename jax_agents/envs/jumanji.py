import importlib
import logging

import jax
import jax.numpy as jnp
import numpy as np
import jumanji
from gymnasium import Env, spaces
from jumanji import specs
from jumanji.types import StepType

logger = logging.getLogger(__name__)

# Generators/reward functions that can be referenced by bare name in environment configs.
_GENERATOR_MODULES = {
    "ToyGenerator": "jumanji.environments.routing.sokoban.generator",
    "SimpleSolveGenerator": "jumanji.environments.routing.sokoban.generator",
    "DeepMindGenerator": "jumanji.environments.routing.sokoban.generator",
    "HuggingFaceDeepMindGenerator": "jumanji.environments.routing.sokoban.generator",
}
_REWARD_MODULES = {
    "DenseReward": "jumanji.environments.routing.sokoban.reward",
    "SparseReward": "jumanji.environments.routing.sokoban.reward",
}


def build_object(spec):
    """
        Resolve a configuration entry into an instance.

    :param spec: either an instance (returned as is), a class name registered in the tables
                 above, or a dict {"__class__": "module.path.ClassName", **kwargs} following
                 the repository's agent-configuration convention.
    :return: the corresponding instance
    """
    if isinstance(spec, dict):
        class_path = spec.get("__class__")
        kwargs = {k: v for k, v in spec.items() if k != "__class__"}
        module_name, _, class_name = class_path.rpartition(".")
        class_ = getattr(importlib.import_module(module_name), class_name)
        return class_(**kwargs)
    if isinstance(spec, str):
        module_name = _GENERATOR_MODULES.get(spec) or _REWARD_MODULES.get(spec)
        if module_name:
            return getattr(importlib.import_module(module_name), spec)()
    return spec


def resolve_make_kwargs(make_kwargs):
    """
        Instantiate the constructor arguments (generator, reward_fn, viewer) of a Jumanji
        environment from their configuration representation.
    """
    make_kwargs = dict(make_kwargs)
    for key in ("generator", "reward_fn", "viewer"):
        if key in make_kwargs:
            make_kwargs[key] = build_object(make_kwargs[key])
    return make_kwargs


def spec_to_space(spec):
    """
        Convert a Jumanji spec to a gymnasium space.
    """
    if isinstance(spec, specs.DiscreteArray):
        return spaces.Discrete(int(spec.num_values))
    if isinstance(spec, specs.MultiDiscreteArray):
        return spaces.MultiDiscrete(spec.num_values)
    if isinstance(spec, specs.BoundedArray):
        dtype = np.dtype(spec.dtype)
        return spaces.Box(low=np.asarray(spec.minimum).item(),
                          high=np.asarray(spec.maximum).item(),
                          shape=spec.shape, dtype=dtype)
    raise TypeError("Unsupported Jumanji spec: {}".format(spec))


class JumanjiEnv(Env):
    """
        A single-environment gymnasium bridge over a Jumanji environment.

        This is the interface used by the repository's usual evaluation loop
        (`Evaluation.run_episodes`, `experiments.py --test`, legacy agents). It converts
        between the gymnasium protocol and Jumanji's functional (state, timestep) protocol.

        Limitations:
        - Only grid-based observations are exposed (e.g. Sokoban's (num_rows, num_cols, 2)
          grid); other observation fields are not part of the observation_space.
        - No rendering: Jumanji environments provide matplotlib viewers, not rgb frames.
          Use `env.unwrapped.jumanji_env.animate(states)` to save a gif.
        - Jumanji has no `truncated` concept: episodes terminate when all boxes are on
          targets or the time limit is reached, reported as `done`.
    """

    metadata = {"render_modes": []}

    def __init__(self, env_id, render_mode=None, **make_kwargs):
        super().__init__()
        self.env_id = env_id
        self.render_mode = render_mode
        self.jumanji_env = jumanji.make(env_id, **resolve_make_kwargs(make_kwargs))
        self.action_space = spec_to_space(self.jumanji_env.action_spec)
        self.observation_space = self._observation_space(self.jumanji_env.observation_spec)
        self._state = None
        self._key = jax.random.PRNGKey(0)
        self.done = False  # Read by the Evaluation loop through env.unwrapped.done

    @staticmethod
    def _observation_space(observation_spec):
        grid_spec = getattr(observation_spec, "grid", None)
        if grid_spec is None:
            raise NotImplementedError(
                "Only grid-based Jumanji observations (e.g. Sokoban) are supported by JumanjiEnv.")
        return spec_to_space(grid_spec)

    @staticmethod
    def _to_numpy_observation(observation):
        return np.asarray(getattr(observation, "grid", observation))

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
        key, self._key = jax.random.split(self._key)
        state, timestep = self.jumanji_env.reset(key)
        self._state = state
        self.done = False
        return self._to_numpy_observation(timestep.observation), {}

    def step(self, action):
        state, timestep = self.jumanji_env.step(self._state, jnp.asarray(action, jnp.int32))
        self._state = state
        terminated = bool(timestep.step_type == StepType.LAST)
        self.done = terminated
        info = {}
        for name, value in (timestep.extras or {}).items():
            value = np.asarray(value)
            info[name] = value.item() if value.ndim == 0 else value
        return self._to_numpy_observation(timestep.observation), float(timestep.reward), terminated, False, info

    def render(self):
        raise NotImplementedError(
            "Jumanji environments do not provide rgb_array rendering. "
            "Use env.unwrapped.jumanji_env.animate(states) to save an animation.")


class VectorizedJumanjiEnv:
    """
        The JAX-native interface used by JAX agents for training: jitted and vectorized
        resets and steps over a batch of environments. All operations are pure functions
        of (states, actions, keys) pytrees.
    """

    def __init__(self, jumanji_env):
        self.jumanji_env = jumanji_env
        self.reset = jax.jit(jax.vmap(jumanji_env.reset))
        self.step = jax.jit(jax.vmap(jumanji_env.step))

    @staticmethod
    def observation(observation):
        """
            Extract the float32 observation grid of a (batch of) Jumanji observation(s).
        """
        return getattr(observation, "grid", observation).astype(jnp.float32)

    def reset_batch(self, keys):
        states, timesteps = self.reset(keys)
        return states, self.observation(timesteps.observation)

    def step_batch(self, states, actions):
        return self.step(states, actions)

    @staticmethod
    def is_done(timesteps):
        return timesteps.step_type == StepType.LAST
