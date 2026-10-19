# TorchRL Integration — Design

Status: **implemented (pilot)** — batch-parallel DQN training with [torchrl](https://github.com/pytorch/rl).

## Goal

Enhance the training efficiency of the repository's value-based agents (DQN first) using torchrl's
batch-parallel data collection, replay buffers and DQN loss modules, while preserving the repository's
patterns: config-driven agents (`__class__` entries), the `AbstractAgent` interface, the `Evaluation`
trainer conventions (metadata, TensorBoard, `saved_models`, `--recover`) and zero changes to legacy code.

## Architecture

```
scripts/torchrl_train.py
    load_environment(env_config) ──► gymnasium env (same configs as everywhere else)
    load_agent(agent_config)    ──► TorchRLDQNAgent(env, config)
        TorchRLEvaluation(Evaluation).train()
            └─ agent.train_loop(evaluation)
                 collector (SyncCollector, direct|process backend, N parallel envs)
                   └─ policy = TensorDictSequential(value_net, QValueModule, EGreedyModule)
                 replay buffer (TensorDictReplayBuffer | PrioritizedReplayBuffer)
                 DQNLoss (double Q, delay_value) + HardUpdate target updater
                 evaluation.save_agent_model / writer / global_step for logging
```

- `rl_agents/agents/torchrl/dqn.py` — `TorchRLDQNAgent(AbstractAgent)`: implements `act` (greedy,
  for `experiments.py --test`), `record` (no-op; training is batch-collected), `save`/`load`
  (`.tar` torch checkpoints, same convention as the pytorch DQN), `seed`, `reset`, `eval`.
- `rl_agents/agents/torchrl/trainer.py` — `TorchRLEvaluation(Evaluation)`: overrides `train()` only;
  inherits metadata, logging, checkpointing, `--recover`.
- The value network is the **same** `model_factory` networks as the pytorch DQN (e.g.
  `MultiLayerPerceptron` [256, 256]): torchrl wraps it in a `TensorDictModule`; the loss uses
  torchrl's `QValueActor` wrapper. Fair comparisons use identical `model` configs.

## Config surface

Agent config keys mirror the pytorch DQN (`model`, `double`, `loss_function`, `optimizer.lr`, `gamma`,
`batch_size`, `memory_capacity`, `target_update`, `exploration`) plus torchrl-specific keys:

```json
{
    "__class__": "<class 'rl_agents.agents.torchrl.dqn.TorchRLDQNAgent'>",
    "model": {"type": "MultiLayerPerceptron", "layers": [256, 256]},
    "double": false, "loss_function": "l2", "optimizer": {"lr": 5e-4},
    "gamma": 0.8, "batch_size": 32, "memory_capacity": 15000, "target_update": 50,
    "prioritized": false, "priority_alpha": 0.6, "priority_beta": 0.4,
    "num_envs": 8, "frames_per_batch": 256, "updates_per_batch": "auto",
    "collector_backend": "batched", "total_steps": 100000, "checkpoint_frequency": 20,
    "exploration": {"method": "EpsilonGreedy", "tau": 6000,
                    "temperature": 1.0, "final_temperature": 0.05},
    "device": "auto"
}
```

- `updates_per_batch: "auto"` = `frames_per_batch // batch_size` (replay ratio 1, same as the legacy
  DQN's one update per collected step); set an integer for a higher replay ratio.
- `collector_backend: "batched"` = `num_envs` environments in a `SerialEnv` inside this process
  (reliable batched policy calls; recommended on macOS); `"direct"` = a single in-process
  environment.
- `exploration.temperature`/`final_temperature` map to `eps_init`/`eps_end`, `tau` to the linear
  annealing horizon of torchrl's `EGreedyModule`.
- `n_steps` is fixed to 1 in this pilot (torchrl's `DQNLoss` uses TD(0) here).

## Efficiency levers (measured)

1. Batch inference during collection — the main win on CPU (10.1× on CartPole, see below).
2. Batched gradient updates (`frames_per_batch`, `batch_size`, replay ratio).
3. GPU batch updates (CUDA machines).
4. Prioritized replay (`"prioritized": true`) — sample efficiency, not yet benchmarked.

## Implementation notes (measured on macOS arm64, CPU)

- Rollouts use an explicit in-process loop over torchrl's `SerialEnv` (batched policy calls, no
  subprocesses): torchrl 0.14's `Collector` process backend and `ParallelEnv` proved unstable on
  macOS (random mid-run hangs, spawn-related worker failures).
- The rollout loop uses `step_and_maybe_reset`, torchrl's designed API for non-stopping rollouts:
  it returns the transition (with the real reward and done flags under `"next"`) and an
  auto-reset tensordict carrying into the next step.
- The environment factory instantiates the environment with its `make_kwargs` (e.g. CarRacing's
  `continuous: false`), applies the repository's environment configuration through `configure`
  (the highway-env convention) and the repository wrappers (frame skip, reward shaping, image
  preprocessing), then exposes it through torchrl's `GymWrapper`.
- The repository's value networks are reused: vector observations are prefixed with a batch-safe
  flattening layer (torchrl calls the policy with unbatched observations for the first rollout
  step), and `ConvolutionalNetwork` models consume the channel-first image observations directly
  (cast to float32) — e.g. CarRacing's `(4, 84, 84)` frame stack from `PreprocessImage`.

## Measured results (M1 Pro, CPU, same model configs and learning rates)

| Benchmark | Legacy DQN | TorchRL DQN | Speedup |
|---|---|---|---|
| CartPole-v0, ~27k env steps | 649 steps/s (42.2s) | 6578 steps/s (8.5s) | **10.1×** |
| CartPole-v0, test score (10 episodes) | 9.0–10.0 | 9.0–10.0 | equal |
| CarRacing-v3 (frameskip 4, 84×84 grayscale), 10k steps | — | ~52 steps/s (3m18s), 36 episodes | — |
| HighwayEnv medium, 20k steps | — | ~79 steps/s (4m19s), test score ≈ 21 | env-bound |
| HighwayEnv medium, 80k steps (legacy) | ~122 steps/s (10m55s), test score ≈ 66 | — | — |

Findings:
- **The throughput win is real when the environment is cheap** (CartPole): batching removes the
  per-step Python/torch overhead that dominates the legacy loop.
- **Expensive simulators bound the speedup on CPU**: highway-env (~12.6ms/step) and CarRacing
  (image preprocessing + Box2D) dominate the per-step cost, so in-process batching helps only
  marginally there; parallel collection (subprocesses) would help on Linux, and GPU batch
  updates apply to CUDA machines.
- Learning results are identical for equal hyperparameters (same networks, same TD(0) loss).
- CarRacing needs far more than 10k steps to drive (test scores around -64…-106); treat the
  current run as a smoke test of the pixel-based pipeline.

## Limitations / notes

- **Action encoding (fixed)**: torchrl's `GymWrapper` defaults to one-hot action encoding, which
  silently decoded the policy's scalar actions to 0 on CarRacing — the root cause of every
  stalled training run. The factory now uses `categorical_action_encoding=True` plus a NumPy-2
  compatibility shim for `read_action`. See
  `lession_and_learn/TORCHRL_CARRACING_FINDINGS.md` for the full investigation.
- torchrl 0.14 API differs from older tutorials (`SyncDataCollector` → `Collector`, `DQNLoss`
  requires `make_value_estimator(ValueEstimators.TD0, gamma=...)`, `QValueModule(action_space=...)`
  takes the string `"categorical"` plus `spec=`).
- `GymWrapper` (wrapping an existing gym env) exposes transitions in the torchrl layout, with
  reward and done flags under `"next"` — hence the `step_and_maybe_reset` rollout loop.
- `--test` and checkpoints reuse the usual conventions (`.tar` torch checkpoints, `--recover`).
- Prioritized replay is wired but not benchmarked yet (`"prioritized": true` + alpha/beta).
