import copy

import gymnasium as gym


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
