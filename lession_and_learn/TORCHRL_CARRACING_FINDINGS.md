# TorchRL CarRacing: the parked car was a silent action-encoding bug

Findings from porting the repository's working CarRacing DQN recipe to the torchrl training stack.

## TL;DR

- Every torchrl CarRacing run (DQN and PPO, any hyperparameters, any network) collapsed to the
  "parked car" baseline (~−94): loss ≈ 0.0003–0.005, Q-values flat at the stall value, zero
  crashes in the replay buffer.
- The root cause was **torchrl's `GymWrapper` defaulting to one-hot action encoding**: our policy
  stored scalar action indices, and torchrl decoded them as one-hot — every action became
  `0` ("coast"). The car literally never drove.
- Fix: `GymWrapper(..., categorical_action_encoding=True)` plus a NumPy-2 compatibility shim for
  torchrl's `read_action` (`int(np.array([3]))` raises on NumPy 2).
- After the fix: the buffer contains crash penalties (−100) and tile bonuses, the loss regime
  matches the SB3 reference (0.1–5 with spikes, not 0.0003), and Q-values differentiate.

## Failure signature (what "broken" looked like)

| Signal | Broken | Healthy (SB3 reference) |
|---|---|---|
| Replay rewards | constant −0.5 | −100 … +15 (crashes + tile bonuses) |
| Episodes | all 250 steps (time limit) | mixed: crashes + time limit |
| Loss after warmup | ~0.0003 | 0.5–4.7 with spikes to 100+ |
| Q at 20k steps | −4, spread 0.03 | +4.5, spread 0.9 |
| Test score | −94 (parked) | +756 @ 400k |

A parked car is *consistent* with everything: constant −0.5 reward is fit perfectly by a constant
Q = r/(1−γ), so the loss collapses to ~0 and no gradient signal remains. Loss ≈ 0 is therefore a
danger sign, not a success sign.

## The bisect path (what was ruled out, in order)

1. **Hyperparameters** — γ (0.95/0.98/0.99), lr, warmup, ε schedule, replay ratio: all collapsed.
2. **Batch mode** — single-env run collapsed identically.
3. **Network capacity** — SB3-scale NatureCNN (1.26M params vs the repo's 42k) collapsed identically.
4. **Loss machinery** — verified torchrl's DQNLoss computes bit-identical values to a manual
   SB3-style target; a fully manual train step inside the torchrl loop collapsed identically.
5. **Exploration machinery** — manual argmax+ε-replace exploration collapsed identically.
6. **The data** — dumping the SB3 reference's replay buffer (via `save_replay_buffer`) showed its
   rewards were −100…+15 while ours were constant −0.5: **the cars behaved differently**.
7. **The smoking gun** — a spy on the innermost gym env showed it received action `0` no matter
   what the policy sent. `torchrl`'s default one-hot action encoding was decoding our scalar
   actions to 0.

## Engineering lessons

- **Verify actions reach the environment** whenever a new env wrapper enters the stack. One spy
  (`inner.step = spy`) answered in minutes what four training runs could not.
- **Keep a reference implementation's dynamics as ground truth.** The SB3 run (checkpoints +
  replay buffer dumps) was used purely as a diagnostic instrument; comparing Q-value spreads,
  loss magnitudes, and buffer contents at the same step counts is what isolated the bug.
- **`agent/loss` is not a health metric.** Near-zero loss means "the Bellman equation is satisfied
  by a useless Q-function"; the episode score is the arbiter.
- **Deliberately rule things out before tuning.** Every "fix the hyperparameters" attempt was
  wasted effort once the bug was in the action path.
- **Checkpoint every signal you might need later** (`save_replay_buffer=True` for SB3, per-run
  checkpoints, exit-code markers in training logs) — post-mortems are impossible without artifacts.
- **Rotating checkpoints for big models** (`keep_checkpoints`): ~20 MB × unbounded count is not
  sustainable; keep the last 3.
- **Launchers need real detachment** (`start_new_session`) and exit-code capture; several runs
  died of SIGKILL (-9) from process-group cleanup / memory pressure, and only the exit-code
  markers told us it was external, not a code bug.
- **torchrl 0.14 API drift is real** (no `SyncDataCollector`, `DQNLoss.make_value_estimator`,
  `QValueModule(action_space="categorical", spec=...)`, transitions with reward/done under
  `"next"`) — older tutorials mislead.
