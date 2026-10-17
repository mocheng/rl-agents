# Lessons and findings

Write-ups from the experiments run in this repository. Each file is self-contained; this index gives
the one-paragraph takeaway.

| document | takeaway |
|---|---|
| [DQN_HIGHWAY_ENV_FINDINGS.md](DQN_HIGHWAY_ENV_FINDINGS.md) | HighwayEnv `env_medium` is saturated for mean reward (~69, median 77.6/80) with a simple MLP + 5-vehicle observation + γ=0.8. Longer training, activations, Huber loss, double/dueling, n-step, larger replay, bigger observations, attention and higher γ were neutral or worse. The remaining 18% crash tail is largely reward-optimal risk; safety only comes at a large speed cost. |
| [SB3_HIGHWAY_ENV_FINDINGS.md](SB3_HIGHWAY_ENV_FINDINGS.md) | Stable-Baselines3 integration: SB3's DQN defaults differ (`net_arch=[64,64]`, Huber loss, global-norm clipping). With the architecture and **MSE loss + element-wise clipping** matched (`MSEDQN`), it reached the best mean (72.8, 16% crash); the controlled MSE − Huber difference was +9.06. Default PPO collapsed to a constant action. |
| [MCTS_SHADOW_ENV_FINDINGS.md](MCTS_SHADOW_ENV_FINDINGS.md) | MCTS planning was being rendered into the live UI: `safe_deepcopy_env` copied `render_mode="human"` into planning copies, so each simulated step created a viewer (10 extra viewers / 349 renders per `plan()` call). Fixed by making all planning copies headless "shadow" environments (`headless=True`), plus not forwarding multi-step plans to the viewer. |
| [GYMNASIUM_UPGRADE.md](GYMNASIUM_UPGRADE.md) | Migration from Gymnasium 0.29.1 / highway-env 1.8.2 to Gymnasium 1.4.0 / highway-env 1.12.1: five code fixes (logger API, moved `capped_cubic_video_schedule`, seeding import, CarRacing-v3, pygame-ce), 31 tests green, and a re-validation showing `env_medium` behaviour is preserved (SB3/MCTS evaluations byte-identical to the old stack). |
| [CARRACING_DQN_FINDINGS.md](CARRACING_DQN_FINDINGS.md) | A DQN "parked car" on CarRacing is a discounted-reward/exploration failure, not a broken algorithm. Frame stacking fixed perception (−92 → +324); state-based stall shaping added +409; frame skip added another +119 with lower variance. Penalizing the *state* (speed < 2 m/s) beats penalizing the do-nothing action (+96) and does not tax legitimate coasting. Best simple recipe: MSEDQN + 4×84×84 grayscale + stall penalty + frame skip. |

## Cross-cutting lessons

- **Diagnose before swapping algorithms.** When an agent finds a degenerate "safe" policy, first ask
  what it can *perceive* (does the observation contain motion/state?) and what its *incentives* are
  under discounting (is the delayed reward visible in value terms?). In both benchmarks, the
  bottleneck was the observation horizon or the reward geometry, not network capacity.
- **Reward shaping should target the pathology, not an action index.** Penalizing a low-speed state
  is cleaner and empirically better than taxing a specific action, which mostly penalizes legitimate
  behaviour.
- **Discounting is a design variable.** A smaller γ does not force progress; it shrinks delayed
  rewards. Frame skip and n-step returns extend the effective horizon far more cheaply than raising γ,
  and pair naturally with target networks.
- **Evaluate on the native reward, with paired seeds.** Shaped training rewards are not comparable
  across arms; report tile coverage/length, crash rate and variance too. A single training seed is
  weak evidence — DQN run-to-run variance is large.
- **Long runs need checkpoints.** Two 4-hour CarRacing runs were killed before their final save; a
  `CheckpointCallback` turned a lost run into a usable model at 400k steps.
