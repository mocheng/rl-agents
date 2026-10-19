import pytest

pytest.importorskip("torchrl")

import numpy as np
import gymnasium as gym

from rl_agents.agents.torchrl.ppo import TorchRLPPOAgent
from rl_agents.agents.torchrl.trainer import TorchRLEvaluation

CONFIG = {
    "model": {"type": "MultiLayerPerceptron", "layers": [32]},
    "gamma": 0.99,
    "lmbda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coeff": 0.01,
    "loss_critic_type": "smooth_l1",
    "optimizer": {"lr": 1e-3},
    "update_epochs": 2,
    "minibatch_size": 32,
    "grad_norm_clipping": 0.5,
    "num_envs": 1,
    "frames_per_batch": 64,
    "collector_backend": "direct",
    "total_steps": 126,
    "checkpoint_frequency": 1,
    "device": "cpu",
}


def make_env():
    return gym.make("CartPole-v1")


def test_act():
    env = make_env()
    agent = TorchRLPPOAgent(env, dict(CONFIG))
    obs, _ = env.reset(seed=0)
    action = agent.act(obs)
    assert isinstance(action, int)
    assert 0 <= action < env.action_space.n


def test_train(tmp_path):
    env = make_env()
    agent = TorchRLPPOAgent(env, dict(CONFIG))
    evaluation = TorchRLEvaluation(env, agent,
                                   directory=tmp_path / "out",
                                   num_episodes=1,
                                   sim_seed=0,
                                   render_mode="headless",
                                   display_agent=False,
                                   display_rewards=False,
                                   close_env=False)
    evaluation.train()
    assert evaluation.global_step > 0
    model_dir = tmp_path / "out" / "saved_models"
    assert any(model_dir.glob("*.tar"))


def test_save_load_roundtrip(tmp_path):
    env = make_env()
    agent = TorchRLPPOAgent(env, dict(CONFIG))
    agent.seed(0)
    agent.reset()
    path = agent.save(str(tmp_path / "model.tar"))

    agent2 = TorchRLPPOAgent(env, dict(CONFIG))
    agent2.seed(1)
    agent2.reset()
    agent2.load(path)
    obs, _ = env.reset(seed=0)
    assert agent.act(obs) == agent2.act(obs)
