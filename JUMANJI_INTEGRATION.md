# Jumanji Integration (JAX-native) — Design

Status: **implemented** — see §Implementation status. Design sections below remain the contract.

This document is the contract for adding [Jumanji](https://github.com/instadeepai/jumanji) environments and
JAX-native training to this repo, following the existing patterns (config-driven envs/agents, factory loading,
`Evaluation` logging/checkpointing) while keeping all JAX code in a separate top-level package so legacy
torch/numpy code is never touched.

## Decisions

| Topic | Decision |
|---|---|
| Training entry point | Zero-touch: separate `scripts/jax_train.py` (same precedent as `scripts/sb3_train.py`). `experiments.py` runs JAX agents only with `--test`. |
| First environment | Sokoban (`Sokoban-v0`), not Snake |
| NN / optimizer stack | flax + optax (Jumanji's own ecosystem) |
| Package layout | Sibling top-level package `jax_agents/`, extractable to its own pip package/repo later |
| Hardware | CPU, CUDA GPU, and Apple Silicon Metal GPU, via a `device` config key |
| Benchmarks | Both documented: `--test`-only for JAX agents, and `--train` via `jax_train.py`; `--processes 1` when JAX is involved (XLA + fork is fragile) |

## Verified Sokoban facts (jumanji 1.1.2)

- Registered id: `Sokoban-v0`; `jumanji.make('Sokoban-v0', **kwargs)` forwards kwargs to the constructor
  (so `generator`, `reward_fn`, `time_limit` can come from the env config's `make_kwargs`).
- Observation: `Observation(grid=(10, 10, 2) uint8, step_count=int32 scalar)`; grid = `[variable_grid, fixed_grid]`
  (agent/boxes layer, walls/targets layer).
- Action: `specs.DiscreteArray(4)`, int32 — `0..3` = Up, Right, Down, Left.
- Reward (default `DenseReward`): `+1` per box placed on a target, `-1` per box removed from a target,
  `-0.1` per step, `+10` bonus when all 4 boxes are on targets.
- Termination: all boxes on targets, or `time_limit` reached (default 120). `step_type == LAST`.
- Generators: default is `HuggingFaceDeepMindGenerator` which **downloads the DeepMind Sokoban dataset from
  HuggingFace on first construction** (cached in `~/.cache/huggingface`). `ToyGenerator` and
  `SimpleSolveGenerator` are offline and used for tests/smoke runs.
- Rendering: matplotlib-based `BoxViewer.render(state)` / `animate(states)` — no `rgb_array`, so
  `RecordVideo` is not supported. Sokoban configs should run with `--no-display`.
- Jumanji requires `python>=3.10` (incl. 3.13) and `jax>=0.4.27`; this repo's Python 3.13 venv is fine.

## Apple Silicon (Metal) support

**Finding (macOS arm64, Python 3.13, October 2026):** the Metal plugin could not be used for training.

- `jax-metal`'s last PyPI release is 0.1.1, built for jax ~0.4.34; it executes basic ops but fails on
  neural-network workloads (`x @ w + b` raises `INTERNAL: Unable to serialize MPS module`, and some
  failures hard-crash the process). It was removed from the venv after testing.
- flax >= 0.10 requires jax >= 0.6, and jumanji < 1.2 requires jax < 0.8 — so on Metal-era jax (0.4.34)
  flax must be <= 0.9.x. The combination jax 0.4.34 + jax-metal 0.1.1 + flax 0.9.0 runs Jumanji resets
  and steps on the Metal device, but not Dense layers.
- Conclusion: CPU is the supported backend on this machine. The `device` config key (`auto|cpu|gpu`)
  still selects CUDA GPUs on Linux, where the full modern stack (jax 0.7.x, flax 0.12) works.

## Tested dependency matrix

| Package | Version |
|---|---|
| Python | 3.13.15 (macOS arm64) |
| jax / jaxlib | 0.4.34 |
| jumanji | 1.1.2 |
| flax | 0.9.0 |
| optax | 0.2.4 |
| chex | 0.1.89 |

Install: `pip install -e ".[jax]"` then pin with the table above if pip resolves newer,
incompatible versions (flax >= 0.10 needs jax >= 0.6; jumanji needs jax < 0.8).

## Layout

```
rl-agents/
├── rl_agents/                    # LEGACY — no imports of jax_agents anywhere
│   ├── agents/  (common, deep_q_network, tree_search, sb3, ...)
│   ├── trainer/evaluation.py
│   └── configuration.py
├── jax_agents/                   # NEW sibling package — all JAX lives here
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py           # gymnasium.register(id="jumanji/Sokoban-v0", ...) on import
│   │   └── jumanji.py            # JumanjiEnv (gym wrapper) + VectorizedJumanjiEnv (jit+vmap)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py               # JaxAgent(AbstractAgent): PRNG keys, device, npz save/load
│   │   ├── networks.py           # flax: MLP + ConvEncoder (Sokoban grid)
│   │   ├── memory.py             # ring buffer (numpy), n-step collapse, one jnp.stack per update
│   │   └── dqn.py                # JaxDQNAgent: rollout + train_step + act
│   └── trainer.py                # JaxEvaluation(Evaluation): overrides train() with the JAX loop
├── scripts/
│   ├── experiments.py            # unchanged; JAX agents work with --test
│   ├── jax_train.py              # NEW: docopt CLI, loads env/agent via the SAME factories
│   └── configs/Sokoban/
│       ├── env.json
│       └── JaxDQNAgent.json
├── tests/jax/                    # pytest.importorskip("jax", "jumanji", "flax", "optax")
│   ├── test_jumanji_env.py
│   ├── test_vectorized_env.py
│   └── test_jax_dqn.py
└── setup.py                      # extras_require={"jax": ["jax", "jumanji", "flax", "optax"]}
```

## Dependency rule (one-way)

`jax_agents` may import only this allow-list from `rl_agents` — all tiny, stable modules with no agent
implementations and no torch:

- `rl_agents.configuration` — `Configurable` (default-config merging)
- `rl_agents.agents.common.abstract` — `AbstractAgent` interface
- `rl_agents.agents.common.memory` — `Transition` namedtuple
- `rl_agents.agents.common.factory` — `load_environment` / `load_agent` (reused by `jax_train.py`)
- `rl_agents.trainer.evaluation` — `Evaluation` (subclassed by `JaxEvaluation`)

`rl_agents` never imports `jax_agents`. Config strings are the only place the two worlds meet:
`"import_module": "jax_agents.envs"` and `"__class__": "<class 'jax_agents.agents.dqn.JaxDQNAgent'>"`.

## Config pattern (unchanged format)

`scripts/configs/Sokoban/env.json`:

```json
{
    "id": "jumanji/Sokoban-v0",
    "import_module": "jax_agents.envs",
    "make_kwargs": {
        "generator": { "__class__": "jumanji.environments.routing.sokoban.generator.ToyGenerator" },
        "time_limit": 120
    }
}
```

(ToyGenerator for offline tests; `HuggingFaceDeepMindGenerator` — with `dataset_name`, `proportion_of_files` —
for real training, accepting the one-time HF download.)

`scripts/configs/Sokoban/JaxDQNAgent.json`:

```json
{
    "__class__": "<class 'jax_agents.agents.dqn.JaxDQNAgent'>",
    "model": { "type": "ConvEncoder", "channels": [64, 128] },
    "gamma": 0.99,
    "learning_rate": 3e-4,
    "batch_size": 256,
    "memory_capacity": 100000,
    "target_update": 500,
    "double": true,
    "num_envs": 64,
    "rollout_steps": 16,
    "updates_per_batch": 4,
    "exploration": { "method": "EpsilonGreedy", "initial": 1.0, "final": 0.01, "tau": 50000 },
    "device": "auto"
}
```

## Module contracts

### `jax_agents/envs/jumanji.py`

`JumanjiEnv(gymnasium.Env)` — single-env bridge for `experiments.py --test` (and any legacy agent):

- `__init__(env_id, **make_kwargs)`: `jumanji.make(...)`; map `specs.DiscreteArray -> spaces.Discrete`,
  `specs.MultiDiscreteArray -> spaces.MultiDiscrete`, `specs.BoundedArray -> spaces.Box`; keep `self.jumanji_env`
  and `self._state`.
- `reset(seed=None, options=None)`: `seed -> jax.random.PRNGKey(seed)`; return `np.asarray(timestep.observation), {}`.
- `step(action)`: `state, timestep = env.step(self._state, action)`; `done = timestep.step_type == StepType.LAST`;
  return `(obs, reward, done, False, {})` — Jumanji has no `truncated`; episode is LAST on solve or time limit.
- `render()`: raise `NotImplementedError` (matplotlib viewer only; use `env.jumanji_env.animate(...)` for gifs).
- Registration on import: `gymnasium.register(id="jumanji/Sokoban-v0", entry_point="jax_agents.envs.jumanji:JumanjiEnv", kwargs={"env_id": "Sokoban-v0"})`.

`VectorizedJumanjiEnv` — the JAX-native interface used by training:

- Holds the raw `jumanji.Environment`; `obs_fn(state) -> jax array` (grid cast to float32 + flatten/conv input).
- `reset(keys) -> (states, obs)`, `step(states, actions) -> (next_states, timesteps)` — both
  `jax.jit(jax.vmap(...))` over the batch axis. Pure, immutable, no sync.

### `jax_agents/agents/base.py` — `JaxAgent(AbstractAgent)`

- `batched = True` (uses the existing attribute convention; `Evaluation` stays untouched).
- Holds `key: PRNGKey`, `device`, `params`/`opt_state` pytrees; `seed(seed)` -> `PRNGKey(seed)`.
- `save(filename)`: npz via `jax.tree_util.tree_map(np.asarray, pytree)` with versioned keys
  (`params/...`, `target_params/...`, `opt_state/...`, `steps`, `global_step`); `load` restores.
  Works with `--recover` from both `jax_train.py` and `experiments.py --test`.
- `act(state)`: jitted greedy single-state inference — this is all `experiments.py --test` needs.
- `record(...)`: no-op (training bypasses per-step recording).

### `jax_agents/agents/dqn.py` — `JaxDQNAgent`

Rollout (the JAX-native core), one jit per config:

```python
carry = (states, keys, episode_returns)
scan over rollout_steps:
    q = apply_fn(params, obs_fn(states))                    # (num_envs, 4)
    greedy = argmax(q); random = randint(subkeys)
    actions = where(uniform(subkeys) < eps, random, greedy) # eps: python float from schedule
    next_states, timesteps = vec_env.step(states, actions)
    done = timesteps.step_type == LAST
    returns = (returns + timesteps.reward) * ~done          # episode stats for logging
    auto-reset done envs with fresh keys/states
output: stacked (obs, action, reward, next_obs, terminal)
```

- `train_step` (jitted): double-DQN TD loss (same math as `deep_q_network/pytorch.py:compute_bellman_residual`)
  under `jax.value_and_grad` + `optax.adam`; hard target-param copy every `target_update` steps.
- Buffer: `memory.py` ring buffer stores numpy arrays (single device sync per rollout, not per step);
  `updates_per_batch` jitted gradient steps per rollout.
- Logging: `writer.add_scalar` for loss, value mean, epsilon; reuse `set_global_step`/`set_writer` as today.

### `jax_agents/trainer.py` — `JaxEvaluation(Evaluation)`

Overrides only `train()`; inherits everything else (metadata json, TensorBoard, best-model tracking,
`save_agent_model`, `--recover`, `after_all_episodes` episode summaries):

```python
class JaxEvaluation(Evaluation):
    def train(self):
        # env/agent already built by the shared factories; grab the raw jumanji env via
        # env.unwrapped.jumanji_env, build VectorizedJumanjiEnv, then:
        for epoch in range(num_epochs):
            transitions, episode_returns, duration = agent.collect_rollout(...)
            agent.update(transitions)
            self.global_step += num_envs * rollout_steps
            agent.set_global_step(self.global_step)
            self.after_all_episodes(...)          # episode summaries + checkpoint schedule
        self.close()
```

### `scripts/jax_train.py`

docopt CLI mirroring `sb3_train.py`: `<environment> <agent> [--steps ...] [--seed ...] [--output ...]`,
loads via the shared `load_environment`/`load_agent` factories, wraps in `JaxEvaluation`, trains,
and exits with the run directory path. `--no-display` implied (no rgb rendering).

## Setup

- `setup.py`: `extras_require={"jax": ["jax>=0.4.27", "jumanji", "flax", "optax"]}` — base install unchanged;
  `find_packages()` picks up `jax_agents/` automatically.
- macOS arm64: additionally `pip install jax-metal` (version-matched to jax).
- Tests: `pytest.importorskip("jax", "jumanji", "flax", "optax")` so the legacy suite never requires JAX.

## Phases

1. **Foundation** — deps extra; `jax_agents/envs/` (gym wrapper + vectorized env + registration); tests with
   `ToyGenerator` (offline). This alone unlocks Sokoban for existing agents (pytorch DQN, SB3, MCTS) via `--test`.
2. **Agent base + trainer** — `JaxAgent`, npz checkpointing, `JaxEvaluation`, `scripts/jax_train.py` end-to-end
   with a trivial agent (random policy) to prove logging/checkpoint/`--recover` plumbing.
3. **JAX DQN** — flax networks (ConvEncoder for the 10×10×2 grid, MLP fallback), buffer, jitted rollout +
   train step; smoke test on ToyGenerator, then real training on the HF DeepMind dataset; verify `--test`
   via `experiments.py` loads the npz checkpoint.
4. **Polish/docs** — benchmark JSON (`--test`), HOWTO/README entries, device notes (CUDA/Metal), XLA+fork
   guidance (`--processes 1`).

## Implementation status

All phases implemented (DQN scope):

- `jax_agents/envs/jumanji.py`: `JumanjiEnv` gym wrapper (registered id `jumanji/Sokoban-v0`) +
  `VectorizedJumanjiEnv` (jitted vmapped reset/step).
- `jax_agents/agents/`: `JaxAgent` base (npz checkpoints via pytree flatten + pickled treedef),
  flax `MLP`/`ConvEncoder`, `ReplayBuffer` (n-step returns, mirroring the pytorch DQN's
  `collapse_n_steps`), `JaxDQNAgent` (jitted scan rollout with auto-reset, jitted double-DQN
  train step, optax adam + grad clipping, epsilon schedule).
- `jax_agents/trainer.py`: `JaxEvaluation(Evaluation)` — overrides `train()` only.
- `scripts/jax_train.py`: docopt CLI mirroring `sb3_train.py`.
- `scripts/configs/Sokoban/`: env + agent configs (ToyGenerator and SimpleSolveGenerator offline
  level sets; `env_simple.json` + `JaxDQNAgent_simple.json` are the fast smoke pair).
- `tests/jax/`: 14 tests, `pytest.importorskip`-guarded; legacy suite untouched (36 tests still pass;
  50 total).
- `setup.py`: `extras_require['jax']`; base install unchanged.

Usage:

```bash
pip install -e ".[jax]"
cd scripts
python jax_train.py configs/Sokoban/env.json configs/Sokoban/JaxDQNAgent.json --steps 200000 --seed 0
python experiments.py evaluate configs/Sokoban/env.json configs/Sokoban/JaxDQNAgent.json --test --recover --episodes 10 --no-display
```

Notes:
- Jumanji environments do not render rgb frames: always pass `--no-display` (or `--render headless`).
- Sokoban is a hard exploration problem: expect to train several hundred thousand steps on the toy levels,
  and millions on the full HuggingFace DeepMind dataset (which downloads on first use — switch the
  `generator` entry in `env.json` to `HuggingFaceDeepMindGenerator` with `dataset_name` and
  `proportion_of_files`).
- The default generator config uses `ToyGenerator` (two offline levels) so training is fully offline.

## Validation findings

- **Plumbing verified end-to-end**: `jax_train.py` trains through `JaxEvaluation` (loss, epsilon,
  episode summaries on TensorBoard; `saved_models/*.npz`; JSON-safe metadata), and
  `experiments.py evaluate ... --test --recover` loads the npz checkpoint through the untouched
  legacy evaluation loop.
- **Learning machinery verified by regression tests** (`tests/jax/test_td_convergence.py`): with a frozen
  target network, the TD update is a regression to known targets — terminal transitions converge to
  the rewards, bootstrapped transitions converge to `r + gamma * max Q_target(s')` (double=False).
- **Observed during Sokoban training**: loss converges; the Q-network learns the reward structure
  (states with boxes on targets are valued near the +10 solve bonus); one-step Q-learning does not
  propagate the box-push reward back to the pre-push state within ~100k steps — the classic
  credit-assignment gap that n-step returns (`n_steps` config) address only partially on this
  hard-exploration problem.
- **DQN does not solve Sokoban here**: consistent with the literature (Jumanji's own baseline is A2C with
  1e7+ steps). Solving Sokoban is future work: an A2C/PPO agent on the same rollout machinery (the
  natural next agent), longer training, or prioritized replay.

## Risks

- **jit recompilation**: fixed `num_envs`/`rollout_steps`/batch shapes; `static_argnames` for spec-derived args.
- **HF dataset download** at first reset: document; ToyGenerator keeps tests offline.
- **Metal float32-only** and occasional CPU fallback of XLA ops: acceptable; `device` config is the escape hatch.
- **XLA + fork** in benchmark multiprocessing: `--processes 1` documented; training goes through `jax_train.py`.
- **Metadata serialization** (`write_metadata` -> `serialize(agent)`): keep `config` JSON-safe, never put jax
  arrays in it (same rule pytorch DQN already follows).

## Future (out of scope now)

- JAX PPO on the same rollout machinery (GAE + clipped loss).
- Jitted MCTS: Jumanji states are already pytrees, so a `jax.jit(jax.vmap(...))` planner could replace the
  python `AbstractPlanner` loop and plug into `AbstractTreeSearchAgent.plan`.
- Extract `jax_agents/` into its own pip package (`rl-agents-jax`) depending on `rl-agents` for the
  config/factory/interface layer.
