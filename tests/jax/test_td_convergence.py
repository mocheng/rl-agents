import pytest

pytest.importorskip("jax")
pytest.importorskip("jumanji")

import numpy as np
import gymnasium as gym

import jax_agents.envs  # noqa: F401  (registers jumanji/Sokoban-v0)
from jax_agents.agents.dqn import JaxDQNAgent

CONFIG = {
    "model": {"type": "MLP", "layers": [16]},
    "gamma": 0.9,
    "learning_rate": 1e-2,
    "grad_norm_clipping": 100.0,
    "batch_size": 128,
    "memory_capacity": 2048,
    "target_update": 1000,
    "double": True,
    "num_envs": 8,
    "rollout_steps": 8,
    "updates_per_batch": 4,
    "total_steps": 1000,
    "n_steps": 1,
    "device": "cpu",
}


def make_agent_and_batch(terminals, double=True):
    import jax
    import jax.numpy as jnp
    env = gym.make("jumanji/Sokoban-v0",
                   generator={"__class__": "jumanji.environments.routing.sokoban.generator.ToyGenerator"},
                   time_limit=120)
    config = dict(CONFIG)
    config["double"] = double
    agent = JaxDQNAgent(env, config)
    agent.seed(0)
    agent.reset()
    agent._make_train_step()
    rng = np.random.default_rng(0)
    batch = {
        "obs": rng.normal(size=(128, 10, 10, 2)).astype(np.float32),
        "actions": rng.integers(0, 4, 128).astype(np.int32),
        "rewards": rng.uniform(-1, 1, 128).astype(np.float32),
        "next_obs": rng.normal(size=(128, 10, 10, 2)).astype(np.float32),
        "terminals": np.asarray(terminals, np.float32),
    }
    return agent, {k: jnp.asarray(v) for k, v in batch.items()}


def train(agent, batch, num_updates):
    losses = []
    for _ in range(num_updates):
        loss, agent.params, agent.opt_state = agent._train_step(
            agent.params, agent.target_params, agent.opt_state, batch)
        losses.append(float(loss))
    return losses


def test_td_step_is_regression_to_known_targets():
    """
        The TD update with a frozen target network is a regression to known
        targets, so the predicted Q-values must converge to them.

        Terminal transitions (terminals=1) have no bootstrap: targets are the
        rewards themselves.
    """
    import jax.numpy as jnp
    agent, batch = make_agent_and_batch(terminals=np.ones(128))
    losses = train(agent, batch, num_updates=300)
    q = agent._forward(agent.params, batch["obs"])[np.arange(128), batch["actions"]]
    expected = batch["rewards"]
    error = float(jnp.mean(jnp.abs(q - expected)))
    assert error < 0.05
    assert losses[-1] < losses[0]


def test_td_step_with_bootstrap_uses_frozen_target():
    """
        Non-terminal transitions bootstrap on the (frozen) target network's
        next-state values: targets are r + gamma * max_a' Q_target(s', a').
        With double=False, the argmax is taken on the frozen target network,
        so the targets are fixed during training.
    """
    import jax.numpy as jnp
    agent, batch = make_agent_and_batch(terminals=np.zeros(128), double=False)
    # Compute the analytic targets once, before any training
    q_next = agent._forward(agent.target_params, batch["next_obs"])
    expected = batch["rewards"] + CONFIG["gamma"] * jnp.max(q_next, axis=-1)
    losses = train(agent, batch, num_updates=500)
    q = agent._forward(agent.params, batch["obs"])[np.arange(128), batch["actions"]]
    error = float(jnp.mean(jnp.abs(q - expected)))
    assert error < 0.1
    assert losses[-1] < losses[0]
