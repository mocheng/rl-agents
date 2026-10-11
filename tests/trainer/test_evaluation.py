from pathlib import Path
import gymnasium as gym
import pytest

from rl_agents.agents.simple.random import RandomUniformAgent
from rl_agents.trainer.evaluation import Evaluation


def test_evaluation(tmpdir):
    env = gym.make('CartPole-v0', render_mode='rgb_array')
    agent = RandomUniformAgent(env)
    evaluation = Evaluation(env,
                            agent,
                            directory=tmpdir,
                            num_episodes=3,
                            display_env=False,
                            display_agent=False,
                            display_rewards=False)
    evaluation.train()
    assert any(['metadata' in file.name for file in evaluation.run_directory.iterdir()])


def test_episodes_are_seeded(tmpdir):
    env = gym.make('CartPole-v0')
    agent = RandomUniformAgent(env)
    evaluation = Evaluation(env,
                            agent,
                            directory=tmpdir,
                            num_episodes=2,
                            sim_seed=42,
                            display_env=False,
                            display_agent=False,
                            display_rewards=False)
    seeds = []
    original_reset = evaluation.wrapped_env.reset

    def recording_reset(*args, **kwargs):
        seeds.append(kwargs.get("seed"))
        return original_reset(*args, **kwargs)

    evaluation.wrapped_env.reset = recording_reset
    evaluation.test()
    assert seeds == [42, 43]


def test_interrupted_training_saves_final_checkpoint(tmpdir):
    pytest.importorskip("torch")
    from rl_agents.agents.deep_q_network.pytorch import DQNAgent

    env = gym.make('CartPole-v0')
    agent = DQNAgent(env)
    evaluation = Evaluation(env,
                            agent,
                            directory=tmpdir,
                            num_episodes=10,
                            display_env=False,
                            display_agent=False,
                            display_rewards=False)

    def interrupted_episodes():
        raise KeyboardInterrupt

    evaluation.run_episodes = interrupted_episodes
    with pytest.raises(KeyboardInterrupt):
        evaluation.train()

    assert (evaluation.run_directory / "checkpoint-final.tar").exists()
    assert (Path(tmpdir) / Evaluation.SAVED_MODELS_FOLDER / "latest.tar").exists()


def test_custom_model_name(tmpdir):
    torch = pytest.importorskip("torch")
    from rl_agents.agents.deep_q_network.pytorch import DQNAgent

    env = gym.make('CartPole-v0')
    agent = DQNAgent(env)
    evaluation = Evaluation(env,
                            agent,
                            directory=tmpdir,
                            num_episodes=1,
                            model_name="my_model",
                            display_env=False,
                            display_agent=False,
                            display_rewards=False)
    evaluation.save_agent_model("final")

    saved_models = Path(tmpdir) / Evaluation.SAVED_MODELS_FOLDER
    assert (saved_models / "my_model.tar").exists()
    assert not (saved_models / "latest.tar").exists()

    # Recover the model by name
    env = gym.make('CartPole-v0')
    recovered_agent = DQNAgent(env)
    Evaluation(env,
               recovered_agent,
               directory=tmpdir,
               num_episodes=1,
               recover=True,
               model_name="my_model",
               display_env=False,
               display_agent=False,
               display_rewards=False)
    for expected, recovered in zip(agent.value_net.parameters(), recovered_agent.value_net.parameters()):
        assert torch.equal(expected, recovered)
