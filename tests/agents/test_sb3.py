import gymnasium as gym
import pytest

pytest.importorskip("stable_baselines3")


def test_algorithm_factory_and_agent(tmp_path):
    from rl_agents.agents.sb3.agent import SB3Agent, algorithm_factory, make_model

    env = gym.make('CartPole-v0')
    config = {"algo": "MSEDQN",
              "policy": "MlpPolicy",
              "model_kwargs": {"buffer_size": 100, "learning_starts": 10},
              "policy_kwargs": {"net_arch": [32, 32]}}

    assert algorithm_factory("MSEDQN") is not None
    assert algorithm_factory("DQN") is not None
    with pytest.raises(ValueError):
        algorithm_factory("NOT_AN_ALGORITHM")

    model = make_model(config, env)
    model.save(tmp_path / "model")

    agent = SB3Agent(env, config)
    path = agent.load(tmp_path / "model.zip")
    assert path.name == "model.zip"

    state, _ = env.reset(seed=0)
    assert agent.plan(state)[0] in range(env.action_space.n)
    agent.eval()
    assert agent.act(state) in range(env.action_space.n)

    saved = agent.save(tmp_path / "again.tar")
    assert saved.name == "again.zip"
