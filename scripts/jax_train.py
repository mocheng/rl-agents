"""Train a JAX agent (from the `jax_agents` package) on a Jumanji environment.

Usage:
  jax_train.py <environment> <agent> [options]
  jax_train.py -h | --help

Options:
  -h --help          Show this screen.
  --steps <count>    Total number of environment steps [default: 200000].
  --seed <seed>      Random seed [default: 0].
  --output <dir>     Output directory [default: out/JAX].
"""
import datetime
from pathlib import Path

from docopt import docopt

from rl_agents.agents.common.factory import load_agent, load_environment
from jax_agents.trainer import JaxEvaluation


def main():
    opts = docopt(__doc__)
    steps = int(opts["--steps"])
    seed = int(opts["--seed"])
    env_config_path = opts["<environment>"]
    agent_config_path = opts["<agent>"]

    env = load_environment(env_config_path, render_mode="headless")
    agent = load_agent(agent_config_path, env)
    agent.update_config({"total_steps": steps})

    output_dir = Path(opts["--output"]) / Path(env_config_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = JaxEvaluation(env,
                               agent,
                               directory=output_dir,
                               num_episodes=1,
                               sim_seed=seed,
                               render_mode="headless",
                               display_agent=False,
                               display_rewards=False,
                               close_env=True)
    print("Training {} for {} steps on {} (seed {}, device {})".format(
        agent.__class__.__name__, steps, env_config_path, seed, agent.device))
    evaluation.train()
    return str(evaluation.run_directory)


if __name__ == "__main__":
    main()
