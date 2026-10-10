# Run an RL agent

From the repository root, activate the virtual environment and install the project:

```bash
source .venv/bin/activate
python -m pip install "gymnasium==0.29.1" "numpy<2" tensorboard
python -m pip install -e .
```

Run a headless CartPole DQN training session:

```bash
cd scripts
SDL_VIDEODRIVER=dummy python experiments.py evaluate \
  configs/CartPoleEnv/env.json \
  configs/CartPoleEnv/DQNAgent.json \
  --train --episodes=200 --no-display
```

Use a smaller episode count for a quick smoke run. Results and checkpoints are written under `scripts/out/`.

Gymnasium may warn that `CartPole-v0` is deprecated; this is the environment currently named by the repository configuration.
