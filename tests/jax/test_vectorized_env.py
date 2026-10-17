import pytest

pytest.importorskip("jax")
pytest.importorskip("jumanji")

import jax
import jax.numpy as jnp
import numpy as np

from jumanji.environments.routing.sokoban import Sokoban
from jumanji.environments.routing.sokoban.generator import ToyGenerator
from jumanji.types import StepType

from jax_agents.envs.jumanji import VectorizedJumanjiEnv


def make_env():
    return Sokoban(generator=ToyGenerator(), time_limit=120)


def test_vectorized_reset():
    env = make_env()
    vec = VectorizedJumanjiEnv(env)
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    states, obs = vec.reset_batch(keys)
    assert obs.shape == (8, 10, 10, 2)
    assert obs.dtype == jnp.float32
    assert states.step_count.shape == (8,)


def test_vectorized_matches_single_environment():
    env = make_env()
    vec = VectorizedJumanjiEnv(env)
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    states, obs = vec.reset_batch(keys)

    single_state, single_timestep = jax.jit(env.reset)(keys[0])
    np.testing.assert_allclose(np.asarray(obs[0]), np.asarray(single_timestep.observation.grid))

    actions = jnp.asarray([0, 1, 2, 3], jnp.int32)
    next_states, timesteps = vec.step_batch(states, actions)
    single_next_state, single_timestep = jax.jit(env.step)(single_state, actions[0])

    np.testing.assert_allclose(
        np.asarray(timesteps.observation.grid[0]),
        np.asarray(single_timestep.observation.grid))
    np.testing.assert_allclose(
        np.asarray(next_states.variable_grid[0]),
        np.asarray(single_next_state.variable_grid))
    np.testing.assert_allclose(np.asarray(timesteps.reward[0]), np.asarray(single_timestep.reward))
    assert bool(vec.is_done(timesteps)[0]) == bool(single_timestep.step_type == StepType.LAST)


def test_vectorized_termination_mask():
    env = make_env()
    vec = VectorizedJumanjiEnv(env)
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    states, _ = vec.reset_batch(keys)
    # Run until every environment is done, or 150 steps
    actions = jnp.zeros(4, jnp.int32)
    done = jnp.zeros(4, bool)
    for _ in range(150):
        states, timesteps = vec.step_batch(states, actions)
        done = done | vec.is_done(timesteps)
        if bool(jnp.all(done)):
            break
    assert bool(jnp.all(done))
