import gymnasium as gym
import pytest

torch = pytest.importorskip("torch")


def test_cartpole():
    from rl_agents.agents.deep_q_network.pytorch import DQNAgent

    env = gym.make('CartPole-v0')
    agent = DQNAgent(env, config=None)

    state, info = env.reset()
    n = 2 * agent.config['batch_size']
    for _ in range(n):
        action = agent.act(state)
        assert action is not None

        next_state, reward, done, truncated, info = env.step(action)
        agent.record(state, action, reward, next_state, done, info)

        if done:
            state, info = env.reset()
        else:
            state = next_state

    assert (len(agent.memory) == n or
            len(agent.memory) == agent.config['memory_capacity'])


class RecordingWriter:
    """Minimal Tensorboard writer stub recording the calls it receives."""

    def __init__(self):
        self.scalars = []
        self.histograms = []

    def add_scalar(self, tag, value, global_step=None):
        self.scalars.append((tag, value, global_step))

    def add_histogram(self, tag, values, global_step=None):
        self.histograms.append((tag, values, global_step))

    def add_graph(self, *args, **kwargs):
        pass


def test_tensorboard_statistics():
    from rl_agents.agents.deep_q_network.pytorch import DQNAgent

    env = gym.make('CartPole-v0')
    agent = DQNAgent(env, config=None)
    agent.config['batch_size'] = 2
    agent.config['statistics_frequency'] = 1
    writer = RecordingWriter()
    agent.set_writer(writer)

    state, info = env.reset()
    for _ in range(10):
        action = agent.act(state)
        next_state, reward, done, truncated, info = env.step(action)
        agent.record(state, action, reward, next_state, done, info)
        state, info = env.reset() if done else (next_state, info)

    histogram_tags = [tag for tag, _, _ in writer.histograms]
    assert any(tag.startswith("agent/weights/") for tag in histogram_tags)
    assert any(tag.startswith("agent/gradients/") for tag in histogram_tags)
    scalar_tags = [tag for tag, _, _ in writer.scalars]
    assert "agent/parameters_norm" in scalar_tags
    assert "agent/gradients_norm" in scalar_tags


def test_no_statistics_when_disabled():
    from rl_agents.agents.deep_q_network.pytorch import DQNAgent

    env = gym.make('CartPole-v0')
    agent = DQNAgent(env, config=None)
    agent.config['batch_size'] = 2
    agent.config['statistics_frequency'] = 0
    writer = RecordingWriter()
    agent.set_writer(writer)

    state, info = env.reset()
    for _ in range(10):
        action = agent.act(state)
        next_state, reward, done, truncated, info = env.step(action)
        agent.record(state, action, reward, next_state, done, info)
        state, info = env.reset() if done else (next_state, info)

    assert not writer.histograms
