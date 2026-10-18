import pytest

pytest.importorskip("torchrl")
pytest.importorskip("highway_env")

import numpy as np
import gymnasium as gym

from rl_agents.agents.torchrl.dqn import TorchRLDQNAgent
from rl_agents.agents.torchrl.trainer import TorchRLEvaluation

CONFIG = {
    "model": {"type": "MultiLayerPerceptron", "layers": [32]},
    "double": True,
    "loss_function": "l2",
    "optimizer": {"lr": 1e-3},
    "gamma": 0.9,
    "batch_size": 32,
    "memory_capacity": 1000,
    "target_update": 5,
    "prioritized": False,
    "num_envs": 1,
    "frames_per_batch": 64,
    "updates_per_batch": "auto",
    "collector_backend": "direct",
    "total_steps": 128,
    "checkpoint_frequency": 1,
    "exploration": {"method": "EpsilonGreedy", "tau": 500,
                    "temperature": 0.5, "final_temperature": 0.05},
    "device": "cpu",
}


def make_env():
    return gym.make("CartPole-v1")


def test_act():
    env = make_env()
    agent = TorchRLDQNAgent(env, dict(CONFIG))
    obs, _ = env.reset(seed=0)
    action = agent.act(obs)
    assert isinstance(action, int)
    assert 0 <= action < env.action_space.n


def test_train(tmp_path):
    env = make_env()
    agent = TorchRLDQNAgent(env, dict(CONFIG))
    evaluation = TorchRLEvaluation(env, agent,
                                   directory=tmp_path / "out",
                                   num_episodes=1,
                                   sim_seed=0,
                                   render_mode="headless",
                                   display_agent=False,
                                   display_rewards=False,
                                   close_env=False)
    evaluation.train()
    assert evaluation.global_step >= CONFIG["total_steps"]
    assert agent.buffer is not None and len(agent.buffer) > 0
    # Checkpoints were written
    model_dir = tmp_path / "out" / "saved_models"
    assert any(model_dir.glob("*.tar"))


def test_save_load_roundtrip(tmp_path):
    env = make_env()
    agent = TorchRLDQNAgent(env, dict(CONFIG))
    agent.seed(0)
    agent.reset()
    path = agent.save(str(tmp_path / "model.tar"))
    assert path.endswith(".tar")

    agent2 = TorchRLDQNAgent(env, dict(CONFIG))
    agent2.seed(1)
    agent2.reset()
    agent2.load(path)
    obs, _ = env.reset(seed=0)
    assert agent.act(obs) == agent2.act(obs)


def test_batched_rollout_collects_frames():
    import torch
    config = dict(CONFIG)
    config["collector_backend"] = "batched"
    config["num_envs"] = 4
    config["frames_per_batch"] = 64
    env = make_env()
    agent = TorchRLDQNAgent(env, config)
    agent.seed(0)
    agent.reset()
    rollout_env = agent._make_rollout_env()
    td = rollout_env.reset()
    for _ in range(64 // 4):
        td = agent.policy(td)
        td = rollout_env.step(td)
    assert td.batch_size == torch.Size([4])
    agent.buffer.extend(td)
    assert len(agent.buffer) == 4


def test_carracing_wrapped_env():
    pytest.importorskip("Box2D")
    from rl_agents.agents.common.wrappers import PreprocessImage
    env = gym.make("CarRacing-v3", continuous=False)
    env = PreprocessImage(env, size=32, grayscale=True, frame_stack=4)
    env.make_kwargs = {"continuous": False}
    env.wrappers_config = [{"type": "PreprocessImage",
                            "kwargs": {"size": 32, "grayscale": True, "frame_stack": 4}}]
    config = dict(CONFIG)
    config["model"] = {"type": "ConvolutionalNetwork"}
    config["collector_backend"] = "batched"
    config["num_envs"] = 2
    agent = TorchRLDQNAgent(env, config)
    agent.seed(0)
    agent.reset()
    rollout_env = agent._make_rollout_env()
    td = rollout_env.reset()
    assert td["observation"].shape == (2, 4, 32, 32)
    td = agent.policy(td)
    td = rollout_env.step(td)
    assert td["next", "reward"].shape[0] == 2
    assert int(td["action"].max()) < env.action_space.n
