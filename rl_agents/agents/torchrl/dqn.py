import logging
import time
import importlib
from pathlib import Path

import numpy as np
import gymnasium as gym

import torch
from gymnasium import spaces
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import Categorical, OneHot, PrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import SerialEnv
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.gym_like import GymLikeEnv
from torchrl.modules import EGreedyModule, QValueModule
from torchrl.objectives import DQNLoss, ValueEstimators
from torchrl.objectives.utils import HardUpdate

from rl_agents.agents.common import wrappers as repo_wrappers
from rl_agents.agents.common.abstract import AbstractAgent
from rl_agents.agents.common.models import model_factory, size_model_config
from rl_agents.agents.torchrl.networks import NatureCNN
from rl_agents.agents.torchrl.trainer import save_rotating_checkpoint
from rl_agents.trainer.statistics import log_training_progress

logger = logging.getLogger(__name__)


def _read_action_numpy2_compat(self, action):
    """
        NumPy 2 compatibility shim for torchrl's GymWrapper.read_action: the original
        converts unbatched actions to size-1 numpy arrays and then calls int(array), which
        raises on NumPy 2. Reshape to a 0-dim array first.
    """
    action = GymLikeEnv.read_action(self, action)
    if (
        isinstance(self.action_spec, (OneHot, Categorical))
        and action.size == 1
        and not self._is_batched
    ):
        if isinstance(action, np.ndarray) and action.ndim == 1:
            action = action.reshape(())
        action = int(action)
    return action


GymWrapper.read_action = _read_action_numpy2_compat


def _create_torchrl_env(env_id, device, import_module=None, env_config=None, make_kwargs=None,
                        wrapper_configs=None):
    """
        Module-level environment factory (picklable). Imports the environment module, instantiates
        the environment with its make kwargs, applies the repository environment configuration
        through `configure` (the highway-env convention) and the repository wrappers (frame skip,
        reward shaping, image preprocessing), then exposes it through the torchrl `GymWrapper`.

        Note: `GymWrapper` follows the transition layout, with the reward under `("next", "reward").
    """
    if import_module:
        importlib.import_module(import_module)
    env = gym.make(env_id, render_mode=None, **(make_kwargs or {}))
    if env_config:
        target = getattr(env, "unwrapped", env)
        if hasattr(target, "configure"):
            target.configure(env_config)
    for wrapper_config in wrapper_configs or []:
        wrapper_class = getattr(gym.wrappers, wrapper_config["type"], None)
        if wrapper_class is None:
            wrapper_class = getattr(repo_wrappers, wrapper_config["type"])
        env = wrapper_class(env, **wrapper_config.get("kwargs", {}))
    env.reset()  # apply the configuration and initialize wrapper state (e.g. frame stacking)
    # categorical_action_encoding=True: actions are scalar indices, not one-hot vectors.
    # Without it, torchrl decodes scalar actions to 0 and the agent can never act!
    return GymWrapper(env, device=device, categorical_action_encoding=True)


class _Flatten(torch.nn.Module):
    """
        Flatten an observation to a batch of vectors: the last `obs_dims` dimensions are
        collapsed, everything before is treated as the batch dimension. Works for both
        single unbatched observations (e.g. shape (5, 5) with obs_dims=2 -> (1, 25)) and
        batches (shape (B, 5, 5) -> (B, 25)).
    """

    def __init__(self, obs_dims):
        super().__init__()
        self.obs_dims = obs_dims

    def forward(self, x):
        x = x.float()
        if self.obs_dims <= 1:
            return x.reshape(-1, x.shape[-1])
        return x.reshape(-1, int(np.prod(x.shape[-self.obs_dims:])))


class _CastFloat(torch.nn.Module):
    """
        Cast a (batch of) observation(s) to float32, for convolutional networks over
        channel-first image observations.
    """

    def forward(self, x):
        return x.float()


class TorchRLDQNAgent(AbstractAgent):
    """
        A Deep Q-Network trained with torchrl's batched environments, replay buffers and
        DQN loss modules.

        The value network is built with the repository's `model_factory` (same networks and model
        configuration as the pytorch DQN), wrapped in torchrl modules. The agent implements the
        repository's agent interface, so `experiments.py evaluate ... --test --recover` runs
        unchanged, while training goes through `scripts/torchrl_train.py` and
        `rl_agents.agents.torchrl.trainer.TorchRLEvaluation`.
    """

    def __init__(self, env, config=None):
        super().__init__(config)
        self.env = env
        assert isinstance(env.action_space, spaces.Discrete), \
            "Only compatible with Discrete action spaces."
        size_model_config(self.env, self.config["model"])
        self.device = self._resolve_device(self.config["device"])
        self.egreedy = None
        self.value_net = None
        self.value_module = None
        self.policy = None
        self.loss_module = None
        self.optimizer = None
        self.updater = None
        self.buffer = None
        self._initialized = False

    @classmethod
    def default_config(cls):
        return dict(model=dict(type="MultiLayerPerceptron", layers=[256, 256]),
                    double=False,
                    loss_function="l2",
                    optimizer=dict(lr=5e-4),
                    gamma=0.8,
                    n_steps=1,
                    batch_size=32,
                    memory_capacity=15000,
                    target_update=50,
                    learning_starts=0,
                    gradient_clipping="none",  # "elementwise" (MSEDQN), a max-norm value, or none
                    prioritized=False,
                    priority_alpha=0.6,
                    priority_beta=0.4,
                    num_envs=8,
                    frames_per_batch=256,
                    updates_per_batch="auto",
                    collector_backend="batched",
                    total_steps=100000,
                    checkpoint_frequency=20,
                    keep_checkpoints=3,
                    diagnostics_frequency=0,
                    exploration=dict(method="EpsilonGreedy",
                                     tau=6000,
                                     temperature=1.0,
                                     final_temperature=0.05),
                    device="auto")

    @staticmethod
    def _resolve_device(name):
        if name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(name)

    def _build(self):
        """
            Build the value network, policy, replay buffer and loss modules.
        """
        if self.config["n_steps"] != 1:
            raise ValueError("Only n_steps=1 is supported by TorchRLDQNAgent (TD(0)).")
        exploration = self.config["exploration"]
        if exploration.get("method") != "EpsilonGreedy":
            raise ValueError("Only the EpsilonGreedy exploration method is supported.")

        action_spec = Categorical(n=int(self.env.action_space.n), device=self.device)
        # The repository models expect a batch of vectors; torchrl collectors may call the
        # policy with single unbatched observations, so prepend a batch-safe flattening layer
        # and disable the model's own reshape. Convolutional networks instead consume the
        # channel-first image observations directly (cast to float32).
        model_config = dict(self.config["model"])
        if model_config["type"] == "NatureCNN":
            self.value_net = NatureCNN(in_channels=self.env.observation_space.shape[0],
                                       out_features=self.env.action_space.n)
        elif model_config["type"] == "ConvolutionalNetwork":
            self.value_net = torch.nn.Sequential(_CastFloat(), model_factory(model_config))
        else:
            model_config["reshape"] = False
            obs_shape = self.env.observation_space.shape
            self.value_net = torch.nn.Sequential(_Flatten(len(obs_shape)), model_factory(model_config))
        self.value_module = TensorDictModule(self.value_net,
                                             in_keys=["observation"],
                                             out_keys=["action_value"])
        self.egreedy = EGreedyModule(spec=action_spec,
                                     eps_init=float(exploration.get("temperature", 1.0)),
                                     eps_end=float(exploration.get("final_temperature", 0.05)),
                                     annealing_num_steps=int(exploration.get("tau", 5000)))
        self.policy = TensorDictSequential(
            self.value_module,
            QValueModule(action_space="categorical", spec=action_spec),
            self.egreedy)

        self.loss_module = DQNLoss(self.value_net,
                                   loss_function=self.config["loss_function"],
                                   delay_value=True,
                                   double_dqn=bool(self.config["double"]),
                                   action_space="categorical")
        self.loss_module.make_value_estimator(ValueEstimators.TD0, gamma=float(self.config["gamma"]))
        self.optimizer = torch.optim.Adam(self.loss_module.parameters(),
                                          lr=float(self.config["optimizer"]["lr"]))
        self.updater = HardUpdate(self.loss_module,
                                  value_network_update_interval=int(self.config["target_update"]))

        storage = LazyMemmapStorage(max_size=int(self.config["memory_capacity"]))
        if self.config["prioritized"]:
            self.buffer = PrioritizedReplayBuffer(
                alpha=float(self.config["priority_alpha"]),
                beta=float(self.config["priority_beta"]),
                storage=storage)
        else:
            self.buffer = TensorDictReplayBuffer(storage=storage)

    def _make_rollout_env(self):
        """
            Build the batched rollout environment.

            - "batched": `num_envs` environments wrapped in a `SerialEnv` inside this process,
              with batched policy calls (recommended).
            - "direct": a single in-process environment.

            Note: torchrl's `Collector` and subprocess-based envs (`ParallelEnv`) proved unstable
            on macOS, so rollouts are collected with an explicit in-process loop over `SerialEnv`.
        """
        backend = self.config["collector_backend"]
        env_id = self.env.unwrapped.spec.id
        import_module = getattr(self.env, "import_module", None)
        env_config = dict(getattr(self.env.unwrapped, "config", None) or {})
        make_kwargs = dict(getattr(self.env, "make_kwargs", None) or {})
        wrapper_configs = getattr(self.env, "wrappers_config", None)
        device = str(self.device) if not self.device.type.startswith("cuda") else self.device
        env_factory = lambda: _create_torchrl_env(env_id, device, import_module, env_config,
                                                  make_kwargs, wrapper_configs)
        if backend == "batched":
            num_envs = int(self.config["num_envs"])
            if num_envs > 1:
                return SerialEnv(num_envs, env_factory)
            logger.info("Collector backend 'batched' requires num_envs > 1; using a single environment.")
        elif backend != "direct":
            raise ValueError("Unknown collector_backend '{}'. Choose from: batched, direct".format(backend))
        return env_factory()

    def _clip_gradients(self):
        """
            Gradient clipping, mirroring the repository's DQN variants: "elementwise" clamps
            each gradient to [-1, 1] (the MSEDQN recipe), a numeric value clips the global
            gradient norm, and "none" disables clipping.
        """
        clipping = self.config["gradient_clipping"]
        if clipping in (None, "none", 0, "0"):
            return
        if clipping == "elementwise":
            for param in self.loss_module.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            return
        torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), float(clipping))

    def _updates_per_batch(self):
        updates = self.config["updates_per_batch"]
        if updates == "auto":
            return max(1, int(self.config["frames_per_batch"]) // int(self.config["batch_size"]))
        return int(updates)

    def _epsilon(self, step):
        """
            Linear epsilon schedule over collected frames, mirroring the collector's annealing.
        """
        exploration = self.config["exploration"]
        initial = float(exploration.get("temperature", 1.0))
        final = float(exploration.get("final_temperature", 0.05))
        tau = int(exploration.get("tau", 5000))
        progress = min(1.0, step / tau)
        return initial + (final - initial) * progress

    def reset(self):
        """
            Build the training stack on first use (after seeding, for reproducibility).
            No-op if the model has already been initialized or loaded.
        """
        if self._initialized:
            return
        self._build()
        self._initialized = True

    def seed(self, seed=None):
        self.seed_value = seed
        if seed is not None:
            torch.manual_seed(seed)
        return [seed]

    def eval(self):
        self.training = False

    def act(self, state):
        """
            Pick a greedy action for a single state, for evaluation.
        """
        if not self._initialized:
            self.reset()
        with torch.no_grad():
            q = self.value_net(torch.tensor([state], dtype=torch.float, device=self.device))
        return int(q.argmax(1).item())

    def record(self, state, action, reward, next_state, done, info):
        # Training is handled by the agent's own batch-collection loop.
        pass

    def _log_diagnostics(self, batch):
        """
            Log Q-value and TD-target statistics of a sampled batch, to watch the value
            function dynamics during training.
        """
        if not self.config["diagnostics_frequency"] or \
                batch % int(self.config["diagnostics_frequency"]) != 0:
            return
        if len(self.buffer) < int(self.config["batch_size"]):
            return
        sample = self.buffer.sample(int(self.config["batch_size"]))
        with torch.no_grad():
            q = self.value_net(sample["observation"])                     # (B, A)
            q_next = self.value_net(sample["next"]["observation"])        # (B, A)
            rewards = sample["next"]["reward"].reshape(-1)
            done = sample["next"]["done"].reshape(-1).float()
            target = rewards + float(self.config["gamma"]) * q_next.max(-1).values * (1 - done)
            logger.info("[diag] q_mean={:.3f} q_spread={:.3f} target_mean={:.3f} "
                        "reward_mean={:.3f} done_frac={:.3f}".format(
                            float(q.mean()), float((q.max(-1).values - q.min(-1).values).mean()),
                            float(target.mean()), float(rewards.mean()), float(done.mean())))

    def train_loop(self, evaluation):
        """
            Train the agent: collect batched rollouts with an explicit in-process loop, update
            the Q-network, and report through the evaluation object (TensorBoard, checkpoints,
            global step).
        """
        if not self._initialized:
            self.reset()
        self.training = True
        rollout_env = self._make_rollout_env()
        num_envs = int(self.config["num_envs"]) if self.config["collector_backend"] == "batched" else 1
        frames_per_batch = int(self.config["frames_per_batch"])
        steps_per_batch = max(1, frames_per_batch // num_envs)
        frames_per_batch = steps_per_batch * num_envs
        num_batches = int(self.config["total_steps"]) // frames_per_batch
        checkpoint_frequency = int(self.config["checkpoint_frequency"])

        # Number of completed episodes, counted across the vectorized rollouts
        total_episodes = 0
        train_start = time.time()

        td = rollout_env.reset()
        for batch in range(num_batches):
            start = time.time()
            rollout = []
            for _ in range(steps_per_batch):
                td = self.policy(td)
                # step_and_maybe_reset returns the transition (with the real reward and done
                # flags under "next") and an auto-reset tensordict carrying into the next step.
                step_td, td = rollout_env.step_and_maybe_reset(td)
                rollout.append(step_td)
                total_episodes += int(step_td.get(("next", "done")).sum())
            batch_td = torch.stack(rollout, dim=0).reshape(-1)
            self.buffer.extend(batch_td)

            losses = []
            for _ in range(self._updates_per_batch()):
                if len(self.buffer) < int(self.config["batch_size"]):
                    break
                if len(self.buffer) < int(self.config["learning_starts"]):
                    break  # replay warmup: collect transitions before the first update
                sample = self.buffer.sample(int(self.config["batch_size"]))
                result = self.loss_module(sample)
                loss = result.get("loss")
                loss.backward()
                self._clip_gradients()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.updater.step()
                if self.config["prioritized"]:
                    self.buffer.update_priorities(sample.get("index"),
                                                  result.get("td_error").detach())
                losses.append(float(loss.detach()))

            duration = time.time() - start
            evaluation.global_step += frames_per_batch
            self.set_global_step(evaluation.global_step)
            if self.writer is not None and losses:
                self.writer.add_scalar("agent/loss", sum(losses) / len(losses), self.global_step)
                self.writer.add_scalar("agent/epsilon", self._epsilon(self.global_step), self.global_step)
                self.writer.add_scalar("agent/env_steps_per_second", frames_per_batch / max(duration, 1e-6),
                                       self.global_step)
                self.writer.add_scalar("agent/buffer_size", len(self.buffer), self.global_step)
            if checkpoint_frequency and batch % checkpoint_frequency == 0:
                save_rotating_checkpoint(evaluation, batch, keep=int(self.config["keep_checkpoints"]))
            log_training_progress(logger, self.__class__.__name__, evaluation.global_step, total_episodes,
                                  loss=sum(losses) / len(losses) if losses else None,
                                  epsilon=self._epsilon(self.global_step),
                                  steps_per_second=frames_per_batch / max(duration, 1e-6),
                                  time=time.time() - train_start)
            self._log_diagnostics(batch)

    def close(self):
        pass

    def save(self, filename):
        """
            Save the value network and optimizer to a torch checkpoint (repository `.tar` convention).
        """
        if not self._initialized:
            self.reset()
        torch.save({"value_net": self.value_net.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, filename)
        return filename

    def load(self, filename):
        """
            Load the value network and optimizer from a torch checkpoint.
        """
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError("No checkpoint found at {}".format(filename))
        if not self._initialized:
            self.reset()
        checkpoint = torch.load(filename, map_location=self.device)
        self.value_net.load_state_dict(checkpoint["value_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        # Refresh the target network parameters with the loaded weights
        self.updater.step()
        return filename
