# Run an RL agent

From the repository root, activate the virtual environment and install the project:

```bash
source .venv/bin/activate
python -m pip install "gymnasium==0.29.1" "numpy<2" tensorboard
python -m pip install -e .
```

Train a CartPole DQN agent without rendering:

```bash
cd scripts
python experiments.py evaluate \
  configs/CartPoleEnv/env.json \
  configs/CartPoleEnv/DQNAgent.json \
  --train --episodes=200 --render headless
```

Evaluate the latest trained model in a live window:

```bash
python experiments.py evaluate \
  configs/CartPoleEnv/env.json \
  configs/CartPoleEnv/DQNAgent.json \
  --test --recover --episodes=5 --render human
```

Use `--render video` to save MP4 files or `--render headless` to evaluate without output. The legacy
`--no-display` option also selects headless mode.

Use `--model-name <name>` to save the trained model as `saved_models/<name>.tar` instead of
`saved_models/latest.tar`. `--recover` loads the model named by `--model-name` (default: `latest`),
and `--recover-from <file>` loads a specific checkpoint, given either as a path or as a filename
relative to the `saved_models` folder.

Use a smaller episode count for a quick smoke run. Results, checkpoints, and videos are written under
`scripts/out/`.

## HighwayEnv

Install the driving environments (keep them pinned to the `gymnasium` version installed above; from
`highway-env>=1.9` the dependency resolves to a `gymnasium>=1.0` that this project does not support):

```bash
python -m pip install "highway-env==1.8.2"
```

Train a DQN agent on the 3-lane highway task:

```bash
cd scripts
python experiments.py evaluate \
  configs/HighwayEnv/env_medium.json \
  configs/HighwayEnv/agents/DQNAgent/dqn.json \
  --train --episodes=600 --render headless
```

`configs/HighwayEnv/env_easy.json` (empty 2-lane road) trains in seconds and is the fastest smoke test.
`configs/HighwayEnv/agents/DQNAgent/ddqn.json` (Double DQN) and `dueling_ddqn.json` (Dueling + Double)
are drop-in replacements for `dqn.json`. The attention model `ego_attention.json` must be paired with
`configs/HighwayEnv/env_obs_attention.json`, which defines the 7-feature per-vehicle observation it
expects.

Replay the latest checkpoint in a live window, or watch the training curves:

```bash
python experiments.py evaluate \
  configs/HighwayEnv/env_medium.json \
  configs/HighwayEnv/agents/DQNAgent/dqn.json \
  --test --recover --episodes=5 --render human

tensorboard --logdir out/HighwayEnv/DQNAgent
```

All summaries of a run share a single x-axis: the number of environment steps executed so far, so
`episode/total_reward`, `exploration/epsilon`, `agent/loss` and `agent/value_mean` can be compared
directly (`episode/index` gives the episode number on that axis).

Gymnasium may warn that `CartPole-v0` is deprecated; this is the environment currently named by the repository configuration.
