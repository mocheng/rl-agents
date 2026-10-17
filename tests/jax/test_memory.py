import pytest

pytest.importorskip("jax")

import numpy as np

from jax_agents.agents.memory import ReplayBuffer


def test_n_step_collapse():
    # Two environments, 4 steps. Env 0: no terminal; env 1: terminal at t=1.
    obs = np.arange(4 * 2 * 4, dtype=np.float32).reshape(4, 2, 2, 2)  # (T, E, 2, 2)
    rewards = np.array([[1.0, 1.0],
                        [2.0, 2.0],
                        [3.0, 3.0],
                        [4.0, 4.0]], np.float32)
    terminals = np.array([[0.0, 0.0],
                          [0.0, 1.0],
                          [0.0, 0.0],
                          [0.0, 0.0]], np.float32)
    actions = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], np.int32)
    next_obs = obs + 100

    buffer = ReplayBuffer(capacity=16, obs_shape=(2, 2), n_steps=3, gamma=0.5)
    buffer.push(obs, actions, rewards, next_obs, terminals)

    collapsed = [buffer.rewards[i] for i in range(buffer.size)]
    # env 0 (no terminal): t=0: 1 + .5*2 + .25*3 = 2.75; t=1: 2 + .5*3 + .25*4 = 4.0
    assert collapsed[0] == pytest.approx(2.75)
    assert collapsed[2] == pytest.approx(4.5)
    # env 1 (terminal at t=1): t=0: 1 + .5*2 = 2.0 (stop after terminal); t=1: 2.0 (terminal itself)
    assert collapsed[1] == pytest.approx(2.0)
    assert collapsed[3] == pytest.approx(2.0)
    # terminal flags collapse correctly
    assert buffer.terminals[1] == 1.0
    assert buffer.terminals[3] == 1.0
    # next_obs of collapsed t=0 in env 0 is next_obs[t=2]
    np.testing.assert_allclose(buffer.next_obs[0], next_obs[2, 0])


def test_n_step_collapse_respects_rollout_end():
    obs = np.zeros((2, 1, 2, 2), np.float32)
    rewards = np.array([[1.0], [2.0]], np.float32)
    terminals = np.zeros((2, 1), np.float32)
    actions = np.zeros((2, 1), np.int32)
    next_obs = obs + 1
    buffer = ReplayBuffer(capacity=8, obs_shape=(2, 2), n_steps=10, gamma=0.99)
    buffer.push(obs, actions, rewards, next_obs, terminals)
    # t=1 has no look-ahead left: return is just its own reward
    assert buffer.rewards[1] == pytest.approx(2.0)
