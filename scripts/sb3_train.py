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
"""
import datetime
import json
from pathlib import Path

from docopt import docopt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_agents.agents.common.factory import load_agent_config, load_environment
from rl_agents.agents.sb3.agent import make_model


def main():
    opts = docopt(__doc__)
    steps = int(opts["--steps"])
    seed = int(opts["--seed"])
    env_config_path = opts["<environment>"]
    agent_config = load_agent_config(opts["<agent>"])

    env = Monitor(load_environment(env_config_path, render_mode="headless"))
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
    print("Training Stable-Baselines3 {} for {} steps on {} (seed {})".format(
        agent_config["algo"], steps, env_config_path, seed))
    model.learn(total_timesteps=steps, progress_bar=False)

    model_path = output_dir / name
    model.save(model_path)
    metadata = dict(environment=env_config_path, agent=agent_config, steps=steps, seed=seed,
                    model=str(model_path.with_suffix(".zip")))
    (output_dir / (name + ".json")).write_text(json.dumps(metadata, indent=4))
    print("Saved model to {}".format(model_path.with_suffix(".zip")))


if __name__ == "__main__":
    main()
