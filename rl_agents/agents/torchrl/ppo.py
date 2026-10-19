import logging
import time
from pathlib import Path

import torch
from torch.distributions import Categorical
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import PPOLoss
from torchrl.objectives.utils import ValueEstimators

from rl_agents.agents.common.models import model_factory, size_model_config
from rl_agents.agents.torchrl.dqn import TorchRLDQNAgent, _CastFloat, _Flatten
from rl_agents.agents.torchrl.networks import NatureCNN
from rl_agents.agents.torchrl.trainer import save_rotating_checkpoint
from rl_agents.trainer.statistics import log_training_progress

logger = logging.getLogger(__name__)


class TorchRLPPOAgent(TorchRLDQNAgent):
    """
        Proximal Policy Optimization trained with torchrl's PPOLoss + GAE on the same batched
        rollout machinery as `TorchRLDQNAgent` (in-process `SerialEnv`, `step_and_maybe_reset`,
        the repository's env recipe replay and model factory).

        Unlike the DQN agent, PPO does not suffer from value-function collapse on environments
        with near-constant rewards (e.g. CarRacing): the policy gradient directly maximizes the
        clipped surrogate objective on the collected rollouts.
    """

    @classmethod
    def default_config(cls):
        return dict(model=dict(type="ConvolutionalNetwork"),
                    critic_model=None,  # defaults to the actor model architecture
                    gamma=0.99,
                    lmbda=0.95,
                    clip_epsilon=0.2,
                    entropy_coeff=0.01,
                    loss_critic_type="smooth_l1",
                    optimizer=dict(lr=3e-4),
                    update_epochs=10,
                    minibatch_size=128,
                    grad_norm_clipping=0.5,
                    num_envs=2,
                    frames_per_batch=1024,
                    collector_backend="batched",
                    total_steps=500000,
                    checkpoint_frequency=10,
                    keep_checkpoints=3,
                    device="auto")

    def _build(self):
        """
            Build the actor (categorical policy over the value of the repository's model),
            the critic (same architecture with a scalar head), and the PPO loss with a GAE
            value estimator.
        """
        model_config = dict(self.config["model"])
        self.actor_net = self._make_network(model_config)

        critic_config = dict(self.config["critic_model"] or self.config["model"])
        size_model_config(self.env, critic_config)
        critic_config["out"] = 1
        self.critic_net = self._make_network(critic_config)

        self.actor = ProbabilisticActor(
            TensorDictModule(self.actor_net, in_keys=["observation"], out_keys=["logits"]),
            in_keys=["logits"], out_keys=["action"],
            distribution_class=Categorical, return_log_prob=True)
        self.critic = ValueOperator(self.critic_net, in_keys=["observation"], out_keys=["state_value"])

        self.loss_module = PPOLoss(self.actor, self.critic,
                                   clip_epsilon=float(self.config["clip_epsilon"]),
                                   entropy_coeff=float(self.config["entropy_coeff"]),
                                   normalize_advantage=True,
                                   loss_critic_type=self.config["loss_critic_type"])
        self.loss_module.make_value_estimator(ValueEstimators.GAE,
                                              gamma=float(self.config["gamma"]),
                                              lmbda=float(self.config["lmbda"]))
        self.optimizer = torch.optim.Adam(self.loss_module.parameters(),
                                          lr=float(self.config["optimizer"]["lr"]))

    def _make_network(self, model_config):
        """
            Build one of the repository's networks with the same observation preprocessing as
            the DQN agent (batch-safe flattening for vector observations, float cast for
            channel-first images).
        """
        if model_config["type"] == "NatureCNN":
            return NatureCNN(in_channels=self.env.observation_space.shape[0],
                             out_features=model_config["out"])
        if model_config["type"] == "ConvolutionalNetwork":
            return torch.nn.Sequential(_CastFloat(), model_factory(model_config))
        model_config["reshape"] = False
        return torch.nn.Sequential(_Flatten(len(self.env.observation_space.shape)),
                                   model_factory(model_config))

    def act(self, state):
        """
            Pick a greedy action for a single state, for evaluation.
        """
        if not self._initialized:
            self.reset()
        with torch.no_grad():
            logits = self.actor_net(torch.tensor([state], dtype=torch.float, device=self.device))
        return int(logits.argmax(1).item())

    def train_loop(self, evaluation):
        """
            Train the agent: collect batched rollouts with the stochastic policy, then run the
            clipped PPO surrogate objective (with GAE) over several epochs of minibatches.
        """
        if not self._initialized:
            self.reset()
        self.training = True
        rollout_env = self._make_rollout_env()
        num_envs = int(self.config["num_envs"]) if self.config["collector_backend"] == "batched" else 1
        steps_per_batch = max(2, int(self.config["frames_per_batch"]) // num_envs)
        # The last transition of each rollout has no bootstrap value and is discarded
        effective_frames = (steps_per_batch - 1) * num_envs
        num_batches = int(self.config["total_steps"]) // effective_frames
        update_epochs = int(self.config["update_epochs"])
        minibatch_size = int(self.config["minibatch_size"])
        grad_clip = float(self.config["grad_norm_clipping"])
        checkpoint_frequency = int(self.config["checkpoint_frequency"])

        total_episodes = 0
        train_start = time.time()
        td = rollout_env.reset()
        for batch in range(num_batches):
            start = time.time()
            rollout = []
            with torch.no_grad():
                for _ in range(steps_per_batch):
                    td = self.actor(td)
                    td = self.critic(td)
                    step_td, td = rollout_env.step_and_maybe_reset(td)
                    rollout.append(step_td)
                    total_episodes += int(step_td.get(("next", "done")).sum())
                for i in range(len(rollout) - 1):
                    rollout[i].set(("next", "state_value"), rollout[i + 1].get("state_value"))
            data = torch.stack(rollout[:-1]).reshape(-1)

            losses = []
            for _ in range(update_epochs):
                for minibatch in data.split(minibatch_size, dim=0):
                    result = self.loss_module(minibatch)
                    loss = result.get("loss_critic") + result.get("loss_objective") + result.get("entropy")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    losses.append(float(loss.detach()))

            duration = time.time() - start
            evaluation.global_step += data.numel()
            self.set_global_step(evaluation.global_step)
            if self.writer is not None and losses:
                self.writer.add_scalar("agent/loss", sum(losses) / len(losses), self.global_step)
                self.writer.add_scalar("agent/env_steps_per_second",
                                       data.numel() / max(duration, 1e-6), self.global_step)
            if checkpoint_frequency and batch % checkpoint_frequency == 0:
                save_rotating_checkpoint(evaluation, batch, keep=int(self.config["keep_checkpoints"]))
            log_training_progress(logger, self.__class__.__name__, evaluation.global_step, total_episodes,
                                  loss=sum(losses) / len(losses) if losses else None,
                                  steps_per_second=data.numel() / max(duration, 1e-6),
                                  time=time.time() - train_start)

    def save(self, filename):
        """
            Save the actor, critic and optimizer to a torch checkpoint (repository `.tar` convention).
        """
        if not self._initialized:
            self.reset()
        torch.save({"actor": self.actor_net.state_dict(),
                    "critic": self.critic_net.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, filename)
        return filename

    def load(self, filename):
        """
            Load the actor, critic and optimizer from a torch checkpoint.
        """
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError("No checkpoint found at {}".format(filename))
        if not self._initialized:
            self.reset()
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor_net.load_state_dict(checkpoint["actor"])
        self.critic_net.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        return filename
