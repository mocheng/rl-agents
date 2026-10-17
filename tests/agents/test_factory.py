import copy

import gymnasium as gym
import numpy as np
import pytest


def test_safe_deepcopy_env_is_headless():
    from rl_agents.agents.common.factory import safe_deepcopy_env

    env = gym.make('CartPole-v0')
    env.unwrapped.render_mode = "human"
    env.unwrapped.viewer = object()

    shadow = safe_deepcopy_env(env)

    assert shadow.unwrapped.render_mode is None
    assert shadow.unwrapped.viewer is None
    # The source environment is left untouched
    assert env.unwrapped.render_mode == "human"
    assert env.unwrapped.viewer is not None


def test_safe_deepcopy_env_can_preserve_rendering():
    from rl_agents.agents.common.factory import safe_deepcopy_env

    env = gym.make('CartPole-v0')
    env.unwrapped.render_mode = "human"
    env.unwrapped.viewer = object()

    shadow = safe_deepcopy_env(env, headless=False)

    # The render mode is preserved, but the viewer is never copied
    assert shadow.unwrapped.render_mode == "human"
    assert shadow.unwrapped.viewer is None


def test_preprocess_env_returns_headless_shadow():
    from rl_agents.agents.common.factory import preprocess_env

    env = gym.make('CartPole-v0')
    env.unwrapped.render_mode = "human"
    env.unwrapped.viewer = object()

    def simplify():
        # Preprocessors typically return a copy of the environment
        shadow = copy.deepcopy(env.unwrapped)
        assert shadow.render_mode == "human"
        return shadow

    env.unwrapped.simplify = simplify
    shadow = preprocess_env(env, [{"method": "simplify"}])

    assert shadow is not env
    assert shadow.render_mode is None
    assert shadow.viewer is None
    # The source environment is left untouched
    assert env.unwrapped.render_mode == "human"
    assert env.unwrapped.viewer is not None


def test_preprocess_env_can_preserve_rendering():
    from rl_agents.agents.common.factory import preprocess_env

    env = gym.make('CartPole-v0')
    env.unwrapped.render_mode = "human"
    env.unwrapped.viewer = object()
    env.unwrapped.simplify = lambda: copy.deepcopy(env.unwrapped)

    shadow = preprocess_env(env, [{"method": "simplify"}], headless=False)

    # The copying of rendering state is left to the preprocessor
    assert shadow.render_mode == "human"
    assert shadow.viewer is not None


def test_load_environment_make_kwargs():
    pytest.importorskip("Box2D")
    from rl_agents.agents.common.factory import load_environment

    env = load_environment({"id": "CarRacing-v3", "make_kwargs": {"continuous": False}},
                           render_mode="headless")
    assert env.action_space.n == 5
    env.close()


def test_float64_action_wrapper():
    pytest.importorskip("Box2D")
    from rl_agents.agents.common.factory import load_environment
    from rl_agents.agents.common.wrappers import Float64ActionWrapper

    env = Float64ActionWrapper(load_environment({"id": "CarRacing-v3"}, render_mode="headless"))
    env.reset(seed=0)
    # Float32 actions are rejected by the box2d bindings; the wrapper converts them to float64
    env.step(np.array([0.0, 0.5, 0.0], dtype=np.float32))
    env.close()


class DummyImageEnv(gym.Env):
    """An image-observation environment used to test image preprocessing."""

    def __init__(self):
        self.observation_space = gym.spaces.Box(0, 255, (96, 96, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(2)
        self.t = 0

    def reset(self, **kwargs):
        self.t = 0
        return np.zeros((96, 96, 3), dtype=np.uint8), {}

    def step(self, action):
        self.t += 1
        observation = np.full((96, 96, 3), self.t % 255, dtype=np.uint8)
        return observation, 0.0, self.t >= 100, False, {}


class DummyRewardEnv(gym.Env):
    """A tiny environment with a constant reward, to test reward wrappers."""

    def __init__(self):
        self.observation_space = gym.spaces.Box(0, 1, (2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(5)

    def reset(self, **kwargs):
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(2, dtype=np.float32), 1.0, False, False, {}


def test_punish_action_reward():
    from rl_agents.agents.common.wrappers import PunishActionReward

    env = PunishActionReward(DummyRewardEnv(), action=0, penalty=0.1)
    env.reset(seed=0)
    _, penalized, *_ = env.step(0)
    env.reset(seed=0)
    _, other, *_ = env.step(1)
    assert penalized == pytest.approx(0.9)
    assert other == pytest.approx(1.0)


def test_frame_skip():
    from rl_agents.agents.common.wrappers import FrameSkip

    env = FrameSkip(DummyRewardEnv(), skip=4)
    env.reset(seed=0)
    _, reward, *_ = env.step(0)
    assert reward == pytest.approx(4.0)


class DummySpeedEnv(DummyRewardEnv):
    """A tiny environment with a settable speed, to test the stall penalty."""

    def __init__(self):
        super().__init__()
        self.speed = 0.0


def test_punish_stall_reward():
    from rl_agents.agents.common.wrappers import PunishStallReward

    base = DummySpeedEnv()
    env = PunishStallReward(base, min_speed=2.0, penalty=0.1, speed_fn=lambda e: e.speed)

    env.reset(seed=0)
    base.speed = 0.5
    _, stalled, *_ = env.step(0)
    assert stalled == pytest.approx(0.9)

    env.reset(seed=0)
    base.speed = 30.0
    _, rolling, *_ = env.step(0)
    assert rolling == pytest.approx(1.0)

    # The penalty can be restricted to a given action
    env.reset(seed=0)
    base.speed = 0.5
    gated = PunishStallReward(base, min_speed=2.0, penalty=0.1, action=0, speed_fn=lambda e: e.speed)
    gated.reset(seed=0)
    _, not_gated_action, *_ = gated.step(1)
    assert not_gated_action == pytest.approx(1.0)


def test_preprocess_image_wrapper():
    from rl_agents.agents.common.wrappers import PreprocessImage

    env = PreprocessImage(DummyImageEnv(), size=84, grayscale=True, frame_stack=4)
    initial, _ = env.reset(seed=0)
    assert env.observation_space.shape == (4, 84, 84)
    assert initial.shape == (4, 84, 84)
    assert initial.dtype == np.uint8

    for _ in range(5):
        env.step(0)
    reset_again, _ = env.reset(seed=0)
    # The frame stack must be cleared on reset
    assert np.array_equal(initial, reset_again)


def test_load_environment_wrappers():
    pytest.importorskip("Box2D")
    from rl_agents.agents.common.factory import load_environment

    env = load_environment({"id": "CarRacing-v3", "make_kwargs": {"continuous": False},
                            "wrappers": [{"type": "PreprocessImage",
                                          "kwargs": {"size": 84, "grayscale": True, "frame_stack": 4}}]},
                           render_mode="headless")
    observation, _ = env.reset(seed=0)
    assert observation.shape == (4, 84, 84)
    env.close()
