# Stable-Baselines3 agents on HighwayEnv

Adding Stable-Baselines3 (SB3) agents to the repository and comparing them with the built-in DQN
and MCTS on `configs/HighwayEnv/env_medium.json`.

## Integration

- `stable-baselines3 2.9.0` (installed with no dependency changes).
- `rl_agents/agents/sb3/agent.py`: `SB3Agent`, exposing an SB3 model through the repository's
  `AbstractAgent` interface for evaluation. `plan`/`act` call `model.predict` (greedy in test
  mode); `load`/`save` translate between the trainer's `.tar` paths and SB3 `.zip` files.
  Training stays inside SB3.
- `rl_agents/agents/sb3/algorithms.py`: `MSEDQN`, an SB3 DQN using an **MSE** temporal-difference
  loss and **element-wise** gradient clipping to `[-1, 1]`, matching the repository's DQN.
- `scripts/sb3_train.py`: trains on a repository environment config (identical MDP) with
  `model.learn()`, saving the model and a metadata JSON to `out/SB3/<environment>/`.
- Configs: `configs/HighwayEnv/agents/SB3Agent/{dqn_mirror,ppo,dqn_mse}.json`.

Evaluation reuses the repository's trainer, so results are paired with the existing baselines:

```bash
python experiments.py evaluate configs/HighwayEnv/env_medium.json \
  configs/HighwayEnv/agents/SB3Agent/dqn_mse.json \
  --test --recover-from out/SB3/env_medium/dqn_mse_500k.zip --seed 1234 --episodes=50 --render headless
```

## Results

All evaluations use 50 paired episodes at `--seed 1234` (MCTS: 20 episodes).

| agent | steps | mean | std | median | crash rate |
|---|---|---|---|---|---|
| repo DQN 2000 ep (previous run) | 160k | 67.15 | — | — | 18% |
| repo DQN 2000 ep (reproduced) | 160k | 54.20 | 26.29 | 66.43 | 52% |
| repo DQN 4000 ep (previous run) | 320k | 69.07 | 18.03 | 77.62 | 18% |
| repo DQN 4000 ep (reproduced) | 320k | 66.13 | 20.53 | 77.76 | 34% |
| SB3 DQN Huber 64x64 | 200k | 60.64 | 22.10 | 73.00 | 38% |
| SB3 DQN Huber 256x256 | 500k | 63.72 | 22.67 | 77.32 | 38% |
| SB3 PPO 64x64 | 200k | 55.22 | 6.64 | 56.10 | 6% |
| SB3 PPO 256x256 | 500k | 53.27 | 26.38 | 68.53 | 38% |
| SB3 MSEDQN 256x256 | 160k | 63.86 | 22.41 | 77.23 | 32% |
| **SB3 MSEDQN 256x256** | **500k** | **72.78** | **13.10** | **78.04** | **16%** |
| MCTS budget 120 | — | 68.33 | 4.80 | 69.13 | 0% |

Paired tests on the same scenarios:

- **MSE − Huber** (both 256x256, 500k steps, only loss/clipping changed):
  **+9.06** (95% CI +2.17…+15.94, t=2.58, 29/50 wins).
- **MSEDQN 160k − repo DQN 2000 ep (reproduced)**: +9.66 (CI +0.87…+18.45, t=2.15).
- **MSEDQN 500k − repo DQN 4000 ep (reproduced)**: +6.64 (CI +0.72…+12.56, t=2.20).
- **MSEDQN 500k − MCTS budget 120** (20 episodes): +6.67 (CI +2.02…+11.32, t=2.81, 18/20 wins).

## Findings

1. **SB3's default DQN is not the same network.** Its default `net_arch` is `[64, 64]`
   (6,149 parameters), not `[256, 256]` (73,733) as in `dqn.json`. `policy_kwargs.net_arch` must be
   set explicitly; doing so raised the DQN median from 73.0 to 77.3.
2. **SB3's default Huber loss causes the crash tail.** Holding architecture and budget fixed,
   switching to MSE + element-wise clipping improved the mean by **+9.06** and reduced crashes from
   38% to 16%. This independently confirms the earlier `DQN_HIGHWAY_ENV_FINDINGS.md` result that
   Huber/smooth-L1 hurts on this task.
3. **Default PPO collapses to a degenerate policy.** With `ent_coef=0.01` and a 64x64 network it
   chose action 4 (SLOWER) for every step of every episode — a stopped car collects the right-lane
   reward (~0.71/step) without the speed reward. With `net_arch=[256,256]` and `gamma=0.8` it stopped
   collapsing but became fast-and-crashy (38% crash), never learning useful lane changes.
4. **MSEDQN is the best SB3 agent**: 72.78 mean, 78.0 median, 16% crashes, std 13.1. It also has
   the best SB3 training curve (`ep_rew_mean=62.1`, `ep_len_mean=65.0`).

## Caveats

- Each arm was trained with a single seed. The repository DQN shows large run-to-run variance:
  two 2000-episode runs scored 67.15 and 54.20 (18% vs 52% crashes), and two 4000-episode runs
  scored 69.07 and 66.13. Cross-run comparisons are therefore optimistic; the controlled
  Huber-vs-MSE test is the strongest evidence, and a multi-seed study is needed before claiming
  MSEDQN beats the repository DQN.
- Training is not fully reproducible in this repository (global `random` in replay sampling, global
  torch RNG for initialization), which is the source of that variance.

## Reproducing

```bash
cd scripts

# Train MSEDQN
python sb3_train.py configs/HighwayEnv/env_medium.json \
  configs/HighwayEnv/agents/SB3Agent/dqn_mse.json \
  --steps 500000 --seed 42 --name dqn_mse_500k --tensorboard

# Evaluate, paired with the other agents
python experiments.py evaluate configs/HighwayEnv/env_medium.json \
  configs/HighwayEnv/agents/SB3Agent/dqn_mse.json \
  --test --recover-from out/SB3/env_medium/dqn_mse_500k.zip \
  --seed 1234 --episodes=50 --render headless
```
