# CarRacing DQN: from a parked car to lap times

Findings from getting a DQN agent to drive the Gymnasium `CarRacing` environment, including the
reward postmortem, the A/B experiments on anti-stall shaping and frame skip, and a comparison with
the [AeneasDQN](https://github.com/AeneasWeiChiHsu/CarRacing-v3-DQN-) reference solution.

## TL;DR

- A DQN that never moves is not a broken algorithm: CarRacing's reward makes "stay still" a safe,
  cheap local optimum for an **unskilled, discounted** learner. The per-frame penalty is a time
  *cost*, not a progress *reward*.
- **Frame stacking** (the agent must see motion) was the first essential fix: reward went from
  **−92** (parked) to **+324**.
- **Anti-stall reward shaping** was the next big lever: **+313** for the action-index version,
  **+409** for a state-based version that penalizes low speed instead of a specific action.
- **Frame skip 4** added another large gain and much lower variance: **+432** over baseline at only
  400k steps.
- AeneasDQN's ~828 average is real, but measured with `domain_randomize=True`; its essential
  ingredients are the standard ones (frame skip, n-step, prioritized replay, double DQN, anti-stall
  shaping, long training) — not the exotic five-branch visual cortex or ten-head ensemble.

## 1. The problem

A plain DQN on frame-stacked grayscale observations learned to survive but **never moved**:
967 of 1000 decisions were brake, speed 0, displacement 0, reward −92.4. This is not the usual
"agent is bad at driving" failure: it is a stable, degenerate policy.

Environment facts (`CarRacing-v3`, discrete actions):

- observation: `(96, 96, 3)` uint8 frames; episode limit 1000 frames;
- actions: `0 = coast straight, 1 = steer left, 2 = steer right, 3 = gas, 4 = brake`
  (all except 3 have zero throttle);
- reward: **−0.1 per frame**, **+1000/N per newly visited tile**, **−100** for leaving the
  playfield (which terminates the episode);
- documented example: finishing in 732 frames gives `1000 − 0.1 × 732 = 926.8`.

## 2. Why the reward produces a stall

The reward is a **scoring function for a policy that already finishes**, not a learning curriculum.
The agent maximizes the *discounted* return `G_t = Σ γ^k r_{t+k+1}`. Three effects combine:

1. **Discounting shrinks the delayed prize.** A lap takes ~500–900 frames; at `γ=0.99` the terminal
   reward is discounted by `γ^700 ≈ 0.0009`, and at `γ=0.95` by `γ^700 ≈ 10^-16`.
2. **The time cost is bounded.** Not moving costs at most −100 total (≈ −4 discounted at γ=0.99),
   while a crash costs −100 *now*.
3. **Exploration and credit assignment.** Reaching the first tiles requires a long, coherent
   action sequence; ε-random driving mostly ends in early crashes, which teaches the network that
   driving is bad. The greedy stall policy then generates only stalling data, so it never discovers
   the +1000.

Discounted values for representative behaviours (`T=1000`, launch tile `b≈6`, lap at 700 frames,
crash at 50 frames with 5% of tiles):

| behaviour | undiscounted | γ=0.95 | γ=0.99 | γ=0.995 | γ=0.999 |
|---|---|---|---|---|---|
| stall | −94 | **+4.0** | −4.0 | −13.9 | −57.2 |
| full lap (700 frames) | +930 | +32.6 | +138.7 | +263.8 | +675.1 |
| early crash (50 frames) | −55 | −9.0 | −63.1 | −80.4 | −97.7 |
| stall, frame skip 4 | — | −2.0 | −30.8 | −51.2 | −82.5 |
| lap, frame skip 4 | — | **+112.3** | +445.9 | +626.8 | +859.6 |

Consequences:

- **A smaller γ makes stalling worse, not better.** At γ=0.95 the discounted value of parking is
  *positive* — the launch tile outweighs the discounted time penalty.
- The fixes that matter are those that make the future visible in value terms: a **larger γ** (with
  the stability machinery it requires), **frame skip** (fewer discount steps; the effective
  per-frame discount becomes `γ^{1/4} ≈ 0.987` for γ=0.95), and **n-step returns** (bootstrap over
  `γ^n`).
- **Shaping** attacks the stall directly, and must target the *state* (low speed / no progress),
  not one action index.

## 3. What we changed in the repository

- `rl_agents/agents/sb3/agent.py`: `SB3Agent`, exposing SB3 models through the repository's
  `AbstractAgent` for evaluation; `algorithms.py`: `MSEDQN` (DQN with MSE loss and element-wise
  gradient clipping, matching the repository's native DQN).
- `scripts/sb3_train.py`: training with `model.learn()` on repository env configs, TensorBoard
  logging, and **checkpointing** every N steps (added after two 4-hour runs were killed before their
  final save; the checkpoints made the lost runs recoverable).
- `rl_agents/agents/common/wrappers.py`: `Float64ActionWrapper` (box2d/NumPy 2 workaround),
  `PreprocessImage` (grayscale + resize + frame stack), `PunishActionReward`, `PunishStallReward`,
  `FrameSkip`.
- `rl_agents/agents/common/factory.py`: env configs can specify `make_kwargs` (e.g.
  `{"continuous": false}`) and a `wrappers` list.
- Configs in `scripts/configs/CarRacingEnv/` and `scripts/configs/CarRacingEnv/agents/SB3Agent/`.

## 4. Results

All numbers are 10 **paired** episodes at `--seed 1234`, evaluated on the **native (unshaped)**
reward so the shaping term does not inflate the score.

| stage | configuration | steps | mean | std | median |
|---|---|---|---|---|---|
| raw RGB DQN | 96×96×3 RGB, no skip | 300k | **−92.4** | — | — |
| + preprocessing | 4×84×84 grayscale | 500k | **+324.2** | 128.1 | 302.9 |
| A: action-0 penalty | + −0.1 on do-nothing | 500k | **+637.0** | 101.9 | 617.0 |
| B: action-0 + frame skip | A + skip 4 | 400k | **+756.4** | 74.6 | 770.0 |
| C: stall-state penalty | −0.1 when speed < 2 m/s | 500k | **+733.1** | 141.3 | 764.9 |

Paired comparisons:

- A − baseline: **+312.8** (95% CI +240.2…+385.3, t=8.45)
- B − baseline: **+432.2** (CI +355.0…+509.4, t=10.97)
- B − A: **+119.4** (CI +42.7…+196.2, t=3.05), with B using **20% fewer steps**
- C − baseline: **+408.9** (CI +303.7…+514.1, t=7.62)
- C − A: **+96.1** (CI +33.4…+158.9, t=3.00)
- B − C: +23.3 (CI −73.7…+120.3, t=0.47, n.s.)

### Does the action penalty make sense?

Arm A's penalty fires on action 0 ("coast straight") regardless of speed. Diagnostics show that was
mostly a tax on legitimate coasting:

- A: 99% of penalized action-0 decisions occurred at speed >10 (mean 40–46), only 1% near
  standstill; the policy still used action 0 for 35–62% of steps.
- C (state-based): action 0 is still used for 18–27% of decisions, 99% of them at speed >10, and
  it is **not** penalized there. C scores **+96.1** above A, with more gas and steering and higher
  tile coverage (92%/64% on seeds 1234/1235).

**Conclusion:** penalizing the *state* (speed < 2 m/s) is a cleaner and significantly better
anti-stall signal than penalizing a control action. It leaves legitimate straight-line coasting
free.

### PPO side note

PPO + CnnPolicy on raw RGB (500k steps) stayed on the track for full episodes but drove slowly
(mean speed 8–18, wobbling steering, scores −16.7 / −15.1 / +30.0). Standard CarRacing PPO needs
multiple millions of steps; this was not pursued further.

## 5. Comparison with AeneasDQN

The reference repository claims an average of 800+ with `domain_randomize=True`. We verified its
1000-episode evaluation CSV: **mean 828.6, median 848.8, std 114.1, max 936.8, min −58.3**.

Its recipe:

- `CarRacing-v3`, `continuous=False`, `domain_randomize=True`; frame skip 4; 4-frame RGB stack
  (`96,96,12`); no grayscale/resize.
- Reward shaping: −0.1 on the do-nothing action (same blunt trick as Arm A).
- Forced forward exploration: first 5 episodes use random actions from {left, right, gas}.
- n-step returns (5), prioritized replay (SumTree, β=0.4) switched on after 500 episodes,
  Double DQN, Huber loss, Adam 2.5e-4, γ=0.95 (bootstrap) / 0.99 (n-step), target sync every
  10 episodes, one gradient update per episode after 5000 transitions, batch 128.
- **20,000 episodes** (~5M decisions / ~20M frames).
- Model: 1×1 conv to 8 channels, then **5 visual branches** (raw, contrast ×2, Sobel + residual,
  contrast + Sobel, Gabor + residual), each through a residual CNN backbone, attention merge over
  the branches, dense MLP `[512,256,128,64]`, and **10 dueling Q-heads** (5 variants × 2) whose
  outputs are averaged.

What matters and what does not:

- The **essential** ingredients are the standard ones: anti-stall shaping, frame skip, n-step,
  prioritized replay, double DQN, and lots of training.
- The five visual branches and ten-head ensemble are expensive and likely not responsible for most
  of the performance. Our much simpler stack (grayscale 4×84×84, MSE-DQN, single CNN) reached
  **733–756** with the core tricks, without domain randomization.
- Their action-0 penalty has the same weakness identified in Arm A: it taxes coasting at speed.
  A state-based stall penalty is the more principled version.

## 6. Conclusions (plain text)

- CarRacing's reward punishes time and rewards coverage, but it does **not** directly reward
  progress, and it pays the big reward only at the end of a long skilled sequence. An untrained,
  discounted learner therefore prefers the safe, bounded cost of standing still.
- The per-frame penalty alone cannot force movement; it only ranks policies that finish. At small
  γ it even makes stalling positive.
- The cure is not a bigger network. It is: (a) let the agent **see motion** (frame stacking),
  (b) make **stalling explicitly unprofitable** (state-based shaping), and (c) **shorten the path to
  the reward** (frame skip, n-step) so the future is not discounted away.
- **State-based shaping beats action-based shaping.** Penalizing low speed is cleaner, more
  defensible, and empirically better than penalizing the do-nothing action.
- Evaluate on the native reward, with paired seeds, and report tile coverage / episode length
  alongside the score. Training variance is large (single-seed results), and long runs need
  checkpointing.

## 7. Best recipe as it stands

MSEDQN (DQN, MSE loss, element-wise gradient clipping) with:

- `PreprocessImage(size=84, grayscale=True, frame_stack=4)`;
- `PunishStallReward(min_speed=2.0, penalty=0.1)`;
- `FrameSkip(skip=4)` (not yet combined with the stall penalty in a single run — Arms B and C were
  trained separately; combining them is the obvious next experiment);
- 500k–1M decisions, ε-greedy, buffer ~30k, `gamma=0.99`, batch 32, update every 4 decisions.

## 8. Caveats

- One training seed per arm; DQN has substantial run-to-run variance.
- Arm B was evaluated at 400k steps (its 500k runs were killed by harness timeouts; the checkpoint
  at 400k was used).
- Our environment uses the default `domain_randomize=False`, while Aeneas trains/evaluates with
  randomization on; the numbers are not directly comparable.
- The absolute scores here are not a benchmark claim: the goal was to understand and fix the
  degenerate stalling behaviour.

## 9. Reproducing

```bash
cd scripts

# Train the best single arm (stall-state penalty)
python sb3_train.py configs/CarRacingEnv/env_discrete_frames_stall_shaped.json \
  configs/CarRacingEnv/agents/SB3Agent/dqn_mse_frames.json \
  --steps 500000 --seed 42 --name dqn_stall_shaped_500k \
  --output out/SB3/CarRacing --tensorboard --checkpoint-freq 50000

# Evaluate on the native, unshaped environment (paired seeds)
python experiments.py evaluate configs/CarRacingEnv/env_discrete_frames.json \
  configs/CarRacingEnv/agents/SB3Agent/dqn_mse_frames.json \
  --test --recover-from out/SB3/CarRacing/env_discrete_frames_stall_shaped/dqn_stall_shaped_500k.zip \
  --episodes=10 --seed 1234 --render headless

# Watch or record
# add: --render human   |   --episodes=3 --render video
```

## 10. Files and commits

- `rl_agents/agents/sb3/{agent,algorithms}.py`, `scripts/sb3_train.py`
- `rl_agents/agents/common/{factory,wrappers}.py`
- `scripts/configs/CarRacingEnv/env*.json`, `.../agents/SB3Agent/{ppo,dqn,dqn_mse,dqn_mse_frames}.json`
- `tests/agents/{test_factory,test_sb3}.py`
- Commits: SB3 integration `39b3aca`; CarRacing configs `7922981`; shaping + frame skip `37c8e70`;
  state-based stall penalty `ab6082b`.
- Trained models live under `scripts/out/SB3/CarRacing/` (gitignored).
