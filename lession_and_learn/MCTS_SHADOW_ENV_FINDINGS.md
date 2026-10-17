# MCTS planning was being rendered in the UI

Investigation of `MCTSAgent` in `--render human` mode on HighwayEnv, where the window appeared to
show the planning process: the ego car trying actions, crashing, and restarting.

## Root cause

Planning itself was correct — `MCTS.plan()` runs every rollout on a deep copy of the environment
(`safe_deepcopy_env`) and never steps the live environment. The problem was that the copies kept
rendering:

1. `safe_deepcopy_env` excluded `viewer` but copied `render_mode`, so planning copies had
   `render_mode="human"` with `viewer=None`.
2. highway-env calls `self.render()` at the end of **every** `step()` when `render_mode == "human"`,
   and `render()` with `viewer=None` creates a new `EnvViewer` and draws to the display. Each
   simulated rollout step was therefore painted into the live window.
3. Measured on a single `agent.plan()` call in human mode:

   | | viewers | renders |
   |---|---|---|
   | before fix | 1 → **11** | 3 → **349** |
   | after fix | 1 → 1 | 3 → 3 |

4. `highway_env 1.8.2`'s `simplify()` is `copy.deepcopy(env)` with distant vehicles pruned (not a
   finite-MDP conversion), so the preprocessed planning state inherited `viewer`/`render_mode` too.
5. A separate visual artifact existed as well: `Evaluation.step` forwarded the whole multi-step plan
   to `viewer.set_agent_action_sequence`, which drew the predicted future as translucent ghost cars.
   Only the executed action should be shown, so that call was removed.

## Fix: headless shadow environments

Planning copies are now headless **shadow environments**:

- `safe_deepcopy_env(obj, headless=True)` (`rl_agents/agents/common/factory.py`): when `headless`,
  `render_mode` is not copied, so simulation copies never render. `headless=False` preserves the
  render mode (the viewer is never copied, by design).
- `preprocess_env(env, configs, headless=True)`: when a preprocessor returns a new environment
  (e.g. `simplify`), it is converted to a headless shadow with `disable_rendering()`
  (`viewer=None`, `render_mode=None`). `headless=False` opts out.
- `Evaluation.step` no longer forwards action sequences to the viewer.

The live environment is untouched: training, `--render human`, `--render video` and video recording
all keep their render mode. Only internal simulation copies are affected. Their consumers are all
simulation-only: tree-search planners (MCTS, OLOP, BRUE, GBOP, MDP-GapE, sparse sampling, …), CEM
candidates, and robust-EPC model environments.

## Verification

```text
viewers/renders before plan: (1, 3) | after plan: (1, 3)   # no planning rendering
default shadow headless: True
opt-out preserves render_mode: human
```

Tests in `tests/agents/test_factory.py`:

- `safe_deepcopy_env` is headless by default and leaves the source env untouched
- `safe_deepcopy_env(headless=False)` preserves `render_mode`, copies no viewer
- `preprocess_env` returns a headless shadow by default, source untouched
- `preprocess_env(headless=False)` leaves rendering state to the preprocessor

Full suite: 27 passed.

## How MCTS works in this repo (why no model file)

MCTS is a planner, not a learner: it has no parameters and no saved model
(`AbstractTreeSearchAgent.save()`/`load()` return `False`). Each action is produced by an online
lookahead search that uses the environment's transition/reward functions as its dynamics model:

1. `Evaluation.step()` → `agent.plan(observation)`.
2. `AbstractTreeSearchAgent.plan()` resets the tree (default `step_strategy: "reset"`), builds the
   planning state with `preprocess_env` (headless shadow), and calls `MCTS.plan(state, observation)`.
3. `MCTS.plan()` runs `episodes` rollouts; `episodes`/`horizon` are allocated from the configured
   `budget` by `OLOP.allocation` (the debug log shows e.g. `10 / 17` during rollouts).
4. Each rollout, on a `safe_deepcopy_env` copy, does UCB-style selection, expansion with the prior
   policy (`random_available` by default), a rollout with the rollout policy, and backpropagation.
5. `get_plan()` follows the best children; only the first action is executed before replanning.

## Practical notes

- Budget is the compute/quality dial: `iterations10/20/50/100` = budgets 60/120/300/600, roughly
  0.4/1/3/6 seconds per decision step on `env_medium`.
- MCTS does not improve with experience; the tree is recomputed from scratch every step.
- Aggregated over the runs in this session: 69 completed episodes, 1 crash (1.4%). `--seed 1234` was
  crash-free over 20 episodes; a crash simply ends the episode and starts the next one.
- `iterations50.json` had a trailing comma that made it unparseable; fixed.
- Run:

  ```bash
  cd scripts
  python experiments.py evaluate \
    configs/HighwayEnv/env_medium.json \
    configs/HighwayEnv/agents/MCTSAgent/iterations10.json \
    --test --episodes=3 --seed 1234 --render human
  ```

## Files changed

- `rl_agents/agents/common/factory.py` — shadow copies (`safe_deepcopy_env`, `preprocess_env`, `disable_rendering`)
- `rl_agents/trainer/evaluation.py` — no longer forwards plans to the viewer
- `scripts/configs/HighwayEnv/agents/MCTSAgent/iterations50.json` — trailing comma fix
- `tests/agents/test_factory.py` — shadow-environment tests
- `tests/trainer/test_evaluation.py` — removed the earlier unclean viewer test
