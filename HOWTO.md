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

Use a smaller episode count for a quick smoke run. Results, checkpoints, and videos are written under
`scripts/out/`.

Gymnasium may warn that `CartPole-v0` is deprecated; this is the environment currently named by the repository configuration.
