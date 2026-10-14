# Gymnasium 1.4 and highway-env 1.12 upgrade

Branch `gymnasium-1.4`: move the repository from the pinned Gymnasium 0.29.1 / highway-env 1.8.2 to
the current Gymnasium 1.4.0 / highway-env 1.12.1.

## Versions

| package | before | after |
|---|---|---|
| gymnasium | 0.29.1 | 1.4.0 |
| highway-env | 1.8.2 | 1.12.1 (requires gymnasium >= 1.0) |
| pygame | 2.6.1 | pygame-ce 2.5.8 (highway-env dependency, drop-in `pygame` module) |
| gymnasium[box2d] | — | box2d-py 2.3.8 + swig 4.5.1 (CarRacing-v3) |

`stable-baselines3 2.9.0` needs no change (supports `gymnasium>=0.29.1,<2.0`).

Install:

```bash
pip uninstall -y pygame pygame-ce
pip install pygame-ce==2.5.8
pip install --no-deps gymnasium==1.4.0 highway-env==1.12.1
```

`--no-deps` avoids pip installing `pygame` next to `pygame-ce` (Gymnasium's metadata requires the
`pygame` distribution, but `pygame-ce` provides the same `pygame` module). `setup.py` now requires
`pygame-ce` so `pip check` is clean.

## Code changes

1. `rl_agents/trainer/logger.py` — Gymnasium 1.x's logger only has `WARN`/`ERROR`/`min_level`; no
   `INFO`, `DEBUG` or `set_level`. The level is now applied with `gym.logger.min_level`, defaults use
   the stdlib `logging` levels.
2. Same `gym.logger` fixes in `scripts/planners_evaluation.py`,
   `scripts/planners_robust_evaluation.py`, `scripts/planners_visualization.py`, and
   `gym.logger.info` → stdlib logging in `scripts/experiments.py`.
3. `rl_agents/trainer/evaluation.py` — `capped_cubic_video_schedule` moved to
   `gymnasium.utils.save_video`.
4. `rl_agents/agents/common/seeding.py` — the old `gym` (OpenAI) seeding import is replaced by
   `gymnasium.utils.seeding`, and the unseeded branch of `np_random` is fixed (it referenced
   `seed_seq` before assignment).
5. CarRacing configs use `CarRacing-v3` (v2 was removed in Gymnasium 1.x).

## Validation

- `python -m pytest tests` → **31 passed**.
- All HighwayEnv configs load with unchanged observation/action spaces (`env_medium` (5,5),
  `env_obs_attention` (15,7), continuous (2,), multi-agent (3,5,5)).
- The repo DQN, SB3 MSEDQN, and MCTS saved models still load and run; MCTS plans on headless shadow
  environments; human/video rendering work (CarRacing-v3 records mp4).
- `pip check` is clean.

## Caveats

- highway-env 1.8.2 → 1.12.1 changes simulation details. The old numeric results
  (`DQN_HIGHWAY_ENV_FINDINGS.md`, `SB3_HIGHWAY_ENV_FINDINGS.md`) were measured on 1.8.2 and are
  **not comparable**: a quick MCTS check on the same seed moved from ~68 to ~74. Re-evaluate or
  retrain before drawing conclusions on this branch.
- The virtual environment is shared: after checking out `master` (Gymnasium 0.29.1 code), the
  upgraded packages must be reverted:

  ```bash
  git checkout master
  pip uninstall -y pygame-ce gymnasium highway-env
  pip install "gymnasium==0.29.1" "highway-env==1.8.2" pygame
  pip install -e .
  ```
