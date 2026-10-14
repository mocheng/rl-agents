import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN


class MSEDQN(DQN):
    """
        A DQN with a mean-squared-error temporal-difference loss and element-wise gradient
        clipping to [-1, 1], matching the DQN implementation of
        `rl_agents.agents.deep_q_network.pytorch.DQNAgent`.

        The default Stable-Baselines3 DQN instead uses a Huber (smooth L1) loss and global
        gradient-norm clipping, which was measured to be substantially worse on HighwayEnv.
    """

    def train(self, gradient_steps, batch_size=100):
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)
            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Mean-squared-error loss
            loss = F.mse_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Element-wise gradient clipping to [-1, 1], as in the original DQN
            for param in self.policy.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
