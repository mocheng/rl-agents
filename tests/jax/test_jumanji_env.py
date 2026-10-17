import pytest

pytest.importorskip("jax")
pytest.importorskip("jumanji")

import numpy as np
import gymnasium as gym

import jax_agents.envs  # noqa: F401  (registers jumanji/Sokoban-v0)


def make_env():
    return gym.make("jumanji/Sokoban-v0",
                    generator={"__class__": "jumanji.environments.routing.sokoban.generator.ToyGenerator"},
                    time_limit=120)


def test_registration_and_specs():
    env = make_env()
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 4
    assert env.observation_space.shape == (10, 10, 2)
    assert env.observation_space.dtype == np.uint8

    obs, info = env.reset(seed=0)
    assert obs.shape == (10, 10, 2)
    assert obs.dtype == np.uint8
    assert not env.unwrapped.done

    obs, reward, done, truncated, info = env.step(0)
    assert obs.shape == (10, 10, 2)
    assert isinstance(reward, float)
    assert done in (True, False)
    assert truncated is False
    assert isinstance(info, dict)


def test_episode_terminates():
    env = make_env()
    env.reset(seed=0)
    steps = 0
    done = False
    while not done and steps < 200:
        _, _, done, _, _ = env.step(steps % 4)
        steps += 1
    # ToyGenerator levels terminate at the latest when the time limit (120) is reached
    assert done
    assert steps <= 120
    assert env.unwrapped.done

    # The environment can be reset after termination
    obs, _ = env.reset(seed=1)
    assert obs.shape == (10, 10, 2)
    assert not env.unwrapped.done


def test_solved_info_reported():
    env = make_env()
    env.reset(seed=0)
    info = {}
    done = False
    steps = 0
    while not done and steps < 120:
        _, _, done, _, info = env.step(steps % 4)
        steps += 1
    assert "prop_correct_boxes" in info or "solved" in info
