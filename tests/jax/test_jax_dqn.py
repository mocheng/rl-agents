import pytest

pytest.importorskip("jax")
pytest.importorskip("jumanji")

import numpy as np
import gymnasium as gym

import jax_agents.envs  # noqa: F401  (registers jumanji/Sokoban-v0)
from jax_agents.agents.dqn import JaxDQNAgent
from jax_agents.trainer import JaxEvaluation

CONFIG = {
    "model": {"type": "MLP", "layers": [64]},
    "gamma": 0.99,
    "learning_rate": 1e-3,
    "grad_norm_clipping": 10.0,
    "batch_size": 64,
    "memory_capacity": 2048,
    "target_update": 10,
    "double": True,
    "num_envs": 8,
    "rollout_steps": 8,
    "updates_per_batch": 2,
    "total_steps": 128,
    "exploration": {"method": "EpsilonGreedy", "initial": 0.5, "final": 0.05, "tau": 100},
    "device": "cpu",
}


def make_env():
    return gym.make("jumanji/Sokoban-v0",
                    generator={"__class__": "jumanji.environments.routing.sokoban.generator.ToyGenerator"},
                    time_limit=120)


def test_act():
    env = make_env()
    agent = JaxDQNAgent(env, dict(CONFIG))
    obs, _ = env.reset(seed=0)
    action = agent.act(obs)
    assert isinstance(action, int)
    assert 0 <= action < env.action_space.n


def test_train(tmp_path):
    env = make_env()
    agent = JaxDQNAgent(env, dict(CONFIG))
    evaluation = JaxEvaluation(env, agent,
                               directory=tmp_path / "out",
                               num_episodes=1,
                               sim_seed=0,
                               render_mode="headless",
                               display_agent=False,
                               display_rewards=False,
                               close_env=False)
    evaluation.train()
    assert evaluation.global_step >= CONFIG["total_steps"]
    assert agent.steps > 0
    assert agent.buffer is not None
    assert len(agent.buffer) > 0
    # Checkpoints were written (after_some_episodes saves the model when training)
    model_dir = tmp_path / "out" / "saved_models"
    assert any(model_dir.glob("*.npz"))


def test_save_load_roundtrip(tmp_path):
    import jax
    env = make_env()
    agent = JaxDQNAgent(env, dict(CONFIG))
    agent.seed(0)
    agent.reset()
    path = agent.save(str(tmp_path / "model.tar"))
    assert path.endswith(".npz")

    agent2 = JaxDQNAgent(env, dict(CONFIG))
    agent2.seed(1)
    agent2.reset()
    assert agent2.steps == 0
    agent2.load(path)
    leaves1 = jax.tree_util.tree_leaves(agent.params)
    leaves2 = jax.tree_util.tree_leaves(agent2.params)
    assert len(leaves1) == len(leaves2)
    for a, b in zip(leaves1, leaves2):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b))
    # Both agents then agree on the greedy action
    obs, _ = env.reset(seed=0)
    assert agent.act(obs) == agent2.act(obs)


def test_requires_jumanji_environment():
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    try:
        JaxDQNAgent(env, dict(CONFIG))
    except ValueError:
        pass
    else:
        raise AssertionError("JaxDQNAgent should require a jumanji-wrapped environment")
