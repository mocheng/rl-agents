"""Train a Stable-Baselines3 agent on a repository environment.

Usage:
  sb3_train.py <environment> <agent> [options]

Options:
  -h --help          Show this screen.
  --steps <count>    Number of training timesteps [default: 200000].
  --seed <seed>      Random seed [default: 0].
  --name <name>      Model file name, without extension [default: latest].
  --output <dir>     Output directory [default: out/SB3].
  --tensorboard      Log training scalars to TensorBoard in the output directory.
  --checkpoint-freq <steps>  Save a checkpoint every N timesteps; 0 disables [default: 50000].
"""
import datetime
import json
import time
from pathlib import Path

from docopt import docopt
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_agents.agents.common.factory import load_agent_config, load_environment
from rl_agents.agents.common.wrappers import Float64ActionWrapper
from rl_agents.agents.sb3.agent import make_model
from rl_agents.trainer.statistics import format_training_progress


class ProgressCallback(BaseCallback):
    """
        Log one training progress line every `frequency` timesteps, in the same format
        as the other training processes.
    """

    def __init__(self, name, frequency=50000, verbose=0):
        super().__init__(verbose)
        self.name = name
        self.frequency = frequency
        self._last_logged = 0
        self._start_time = None

    def _on_training_start(self):
        self._start_time = time.time()

    def _episode_count(self):
        try:
            envs = getattr(self.training_env, "envs", None)
            if envs is None:
                return 0
            return sum(len(env.episode_returns) for env in envs)
        except (AttributeError, TypeError):
            return 0

    def _on_step(self):
        if self.num_timesteps - self._last_logged < self.frequency:
            return True
        self._last_logged = self.num_timesteps
        elapsed = time.time() - self._start_time
        logger_values = self.model.logger.name_to_value if self.model.logger else {}
        print(format_training_progress(self.name, self.num_timesteps, self._episode_count(),
                                       loss=logger_values.get("train/loss"),
                                       epsilon=logger_values.get("train/exploration_rate"),
                                       steps_per_second=self.num_timesteps / max(elapsed, 1e-6)))
        return True


def main():
    opts = docopt(__doc__)
    steps = int(opts["--steps"])
    seed = int(opts["--seed"])
    env_config_path = opts["<environment>"]
    agent_config = load_agent_config(opts["<agent>"])

    env = load_environment(env_config_path, render_mode="headless")
    if isinstance(env.action_space, spaces.Box):
        env = Float64ActionWrapper(env)
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env.seed(seed)

    output_dir = Path(opts["--output"]) / Path(env_config_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    name = opts["--name"]
    if name == "latest":
        name = "{}_{}_{}".format(agent_config["algo"],
                                 datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), seed)

    model = make_model(agent_config, vec_env, seed=seed,
                       tensorboard_log=str(output_dir / "tensorboard") if opts["--tensorboard"] else None)
    checkpoint_freq = int(opts["--checkpoint-freq"])
    callbacks = []
    if checkpoint_freq > 0:
        callbacks.append(CheckpointCallback(save_freq=checkpoint_freq, save_path=str(output_dir),
                                            name_prefix=name, save_replay_buffer=False,
                                            save_vecnormalize=False))
    callbacks.append(ProgressCallback(agent_config["algo"], frequency=max(steps // 20, 1)))
    print("Training Stable-Baselines3 {} for {} steps on {} (seed {})".format(
        agent_config["algo"], steps, env_config_path, seed))
    model.learn(total_timesteps=steps, progress_bar=False, callback=callbacks)

    model_path = output_dir / name
    model.save(model_path)
    metadata = dict(environment=env_config_path, agent=agent_config, steps=steps, seed=seed,
                    model=str(model_path.with_suffix(".zip")))
    (output_dir / (name + ".json")).write_text(json.dumps(metadata, indent=4))
    print("Saved model to {}".format(model_path.with_suffix(".zip")))
    print("Evaluate it with:\n  python experiments.py evaluate {} {} --test "
          "--recover-from {} --episodes=5 --render human".format(
              env_config_path, opts["<agent>"], model_path.with_suffix(".zip")))


if __name__ == "__main__":
    main()
