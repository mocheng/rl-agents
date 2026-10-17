# DQN on HighwayEnv: tuning investigation

Scope: improving `DQNAgent` performance on `configs/HighwayEnv/env_medium.json`
(`highway-v0`, 3 lanes, 15 traffic vehicles, 40 s episodes at `policy_frequency=2` → 80 steps).
Rewards are normalized per step to `[0, 1]`, so the theoretical maximum is **80** per episode.

## Protocol

- Training: 2000 episodes, `--seed 42`, unless noted. Configs differ from
  `configs/HighwayEnv/agents/DQNAgent/dqn.json` by a single factor where possible.
- Evaluation: `--test --recover`, `--seed 1234`, 50 episodes, greedy policy.
  Episodes are paired across models because each episode resets the environment with
  `sim_seed + episode` (fixed in commit `aac095e`); two runs with the same seed are
  byte-identical.
- Crash rate: fraction of episodes with `episode/length < 80`.
- All results below are on the same 50 seeded scenarios.

## Results

### Baseline and training length

| setting | eval mean | crash rate | mean (no crash) |
|---|---|---|---|
| 600 episodes | 39.2 | — | — |
| 2000 episodes | 67.15 | 18% | 76.91 |
| 4000 episodes | 69.07 | 18% | 76.91 |

Training curve was still rising at 600 episodes but plateaus by ~2000
(4000 vs 2000: +1.92, 95% CI −1.5…+5.4, n.s.). The baseline reaches
a median of 77.6/80 (97% of the ceiling); the mean is dragged down by 9/50
episodes that crash early.

### Failed or neutral interventions (paired eval)

| intervention | config | control mean | mean | crash rate | paired diff |
|---|---|---|---|---|---|
| LeakyReLU (600 ep) | `dqn_leaky_relu.json` | 52.0 (ReLU, 600 ep) | 47.0 | — | −5.0 (−12.5…+2.4, n.s.) |
| ELU (600 ep) | `dqn_elu.json` | 52.0 (ReLU, 600 ep) | 49.3 | — | −2.8 (−9.4…+3.9, n.s.) |
| Huber loss (600 ep) | `dqn_smooth_l1.json` | 39.2 (l2, 600 ep) | 23.6 | — | **−15.6** (−22.0…−9.3) |
| Double DQN (4000 ep) | `ddqn.json` | 69.07 | 65.8 | — | −3.3 (−9.0…+2.4, n.s.) |
| Attention, baseline RL | `ego_attention_baseline.json` | 69.07 | 67.8 | 22% | −1.2 (−6.5…+4.0, n.s.) |
| Replay 100k | `dqn_memory.json` | 69.07 | 65.5 | 22% | −3.5 (−9.3…+2.2, n.s.) |
| n-step 3 | `dqn_nsteps.json` | 69.07 | 47.2 | 44% | **−21.9** (−30.0…−13.8) |
| γ = 0.9 | `dqn_gamma09.json` | 69.07 | 43.5 | 70% | **−25.6** |
| γ = 0.95 | `dqn_gamma095.json` | 69.07 | 39.9 | 52% | **−29.2** |

The activation and loss arms were run at a 600-episode budget and are compared with
their own same-budget controls; the remaining arms were run at 2000 episodes and are
compared with the 4000-episode baseline for the final model.

Gradient clipping is already implemented (element-wise clamp to `[-1, 1]` in
`DQNAgent.step_optimizer`); logged gradient norms are ~3 against a ceiling of ~270,
and no non-finite parameters were observed. This was never the bottleneck.

Reviving "dead" ReLU units does not help: the trained ReLU model has 75/256 and
148/256 permanently-zero units, yet LeakyReLU/ELU — which keep gradients flowing —
do not improve performance.

### Safety/speed trade-off

| model | mean | std | crash rate | mean (no crash) |
|---|---|---|---|---|
| baseline MLP (γ=0.8) | 69.07 | 18.0 | 18% | 76.91 |
| tuned MLP (double, γ=0.99, target 512) | 57.51 | 3.08 | 4% | 57.89 |
| attention (double, γ=0.99, target 512) | 55.22 | 6.64 | 6% | 56.73 |

### Observation size and architecture

Increasing the observation to 15 vehicles (`env_medium_obs15.json`) makes a flat
MLP far worse — and reckless:

| model on 15-vehicle observation | mean | crash rate | mean (no crash) |
|---|---|---|---|
| MLP | 32.78 | 92% | 72.15 |
| dueling | 36.57 | 92% | 74.34 |
| ego attention (paper settings) | 55.22 | 6% | 56.73 |

Attention is +22.4 over the MLP on the same observation, confirming that a
permutation-invariant model is the right structure for multi-vehicle input — but it
still does not beat the simple 5-vehicle MLP baseline.

## Conclusions

1. `env_medium` is saturated for mean reward with this reward function. The simple
   MLP + 5 nearest vehicles + γ=0.8 recipe is a strong local optimum (~69 mean,
   ~77.6 median = 97% of the ceiling). Nothing in the standard DQN toolbox
   (capacity, activation, loss, clipping, double, dueling, n-step, replay size,
   observation size, attention, higher γ) improves it.
2. The remaining gap is the crash tail (18%), not average performance. A crash costs
   one normalized −1 step plus the truncated future, which is cheap under γ=0.8 and
   the default reward; the observed policy is likely close to reward-optimal risk.
   Safety can be bought (γ=0.99 + target 512 gives 4–6% crashes) but costs ~25% of
   the speed reward, so the mean drops to ~55–58.
3. Intermediate γ (0.9/0.95) is worse on both axes: more crashes *and* slower, with
   training crash rates (28–44%) much lower than eval (52–70%), indicating a
   generalization failure rather than a risk-pricing fix.
4. Further DQN tuning on this benchmark has negative expected value. Improving
   safety requires changing the objective (reward shaping, disabling reward
   normalization, or constraint-based agents); improving DQN as a general agent
   requires a harder testbed (`IntersectionEnv`, `MergeEnv`, `RoundaboutEnv`,
   `env_multi_agent.json`), where these levers may matter.

## Caveats

- One training seed per arm. Eval comparisons are paired and deterministic, but
  training runs are independent draws: `ReplayMemory.sample` uses the global
  `random` module and network init uses the global torch RNG, so `--seed` does not
  make training fully reproducible.
- Crash rate at eval is measured with a greedy policy; training crash rates are
  higher because of exploration.

## Reproducing

```bash
cd scripts

# Train one arm (example: gamma 0.9)
python experiments.py evaluate configs/HighwayEnv/env_medium.json \
  configs/HighwayEnv/agents/DQNAgent/dqn_gamma09.json \
  --train --episodes=2000 --seed 42 --render headless --model-name g09

# Paired evaluation
python experiments.py evaluate configs/HighwayEnv/env_medium.json \
  configs/HighwayEnv/agents/DQNAgent/dqn_gamma09.json \
  --test --recover --model-name g09 --seed 1234 --episodes=50 --render headless
```

Configs added for this investigation:
`dqn_gamma09`, `dqn_gamma095`, `dqn_memory`, `dqn_nsteps`, `dueling`,
`ego_attention_baseline`, `env_medium_obs15`, plus the earlier `dqn_leaky_relu`,
`dqn_elu`, `dqn_smooth_l1`, `ddqn_tuned` and their benchmark files.
