import numpy as np
import jax.numpy as jnp


class ReplayBuffer:
    """
        A ring buffer of transitions, stored as numpy arrays and converted to JAX arrays
        once per sampled batch (one device transfer per gradient step, not per transition).

        Transitions are pushed as (T, E, ...) rollout arrays (T steps of E parallel
        environments) and collapsed into n-step returns before storage, mirroring
        `rl_agents.agents.common.memory.ReplayMemory.collapse_n_steps`: the return is the
        discounted sum of the next `n_steps` rewards, truncated at episode boundaries.
    """

    def __init__(self, capacity, obs_shape, n_steps=1, gamma=0.99, seed=None):
        self.capacity = int(capacity)
        self.n_steps = int(n_steps)
        self.gamma = float(gamma)
        self.obs = np.zeros((self.capacity,) + tuple(obs_shape), np.float32)
        self.actions = np.zeros(self.capacity, np.int32)
        self.rewards = np.zeros(self.capacity, np.float32)
        self.next_obs = np.zeros((self.capacity,) + tuple(obs_shape), np.float32)
        self.terminals = np.zeros(self.capacity, np.float32)
        self.position = 0
        self.size = 0
        self.np_random = np.random.default_rng(seed)

    def push(self, obs, actions, rewards, next_obs, terminals):
        """
            Push the transitions of a rollout, one n-step transition per (t, env).

        :param obs: array (T, E, *obs_shape) of pre-step observations
        :param actions: array (T, E) of actions
        :param rewards: array (T, E) of rewards
        :param next_obs: array (T, E, *obs_shape) of next observations
        :param terminals: array (T, E) of terminal flags
        """
        obs = np.asarray(obs, np.float32)
        next_obs = np.asarray(next_obs, np.float32)
        actions = np.asarray(actions, np.int32)
        rewards = np.asarray(rewards, np.float32)
        terminals = np.asarray(terminals, np.float32)
        steps, num_envs = obs.shape[0], obs.shape[1]

        collapsed_obs = np.empty((steps * num_envs,) + obs.shape[2:], np.float32)
        collapsed_actions = np.empty(steps * num_envs, np.int32)
        collapsed_rewards = np.empty(steps * num_envs, np.float32)
        collapsed_next_obs = np.empty((steps * num_envs,) + obs.shape[2:], np.float32)
        collapsed_terminals = np.empty(steps * num_envs, np.float32)

        for env in range(num_envs):
            for t in range(steps):
                reward = rewards[t, env]
                next_state = next_obs[t, env]
                done = terminals[t, env]
                discount = 1.0
                for k in range(1, self.n_steps):
                    if t + k >= steps or done:
                        break
                    discount *= self.gamma
                    reward += discount * rewards[t + k, env]
                    next_state = next_obs[t + k, env]
                    done = terminals[t + k, env]
                index = t * num_envs + env
                collapsed_obs[index] = obs[t, env]
                collapsed_actions[index] = actions[t, env]
                collapsed_rewards[index] = reward
                collapsed_next_obs[index] = next_state
                collapsed_terminals[index] = done

        self.push_collapsed(collapsed_obs, collapsed_actions, collapsed_rewards,
                            collapsed_next_obs, collapsed_terminals)

    def push_collapsed(self, obs, actions, rewards, next_obs, terminals):
        """
            Store already-collapsed transitions (T*E, ...) in the ring buffer.
        """
        n = len(obs)
        if n >= self.capacity:
            obs = obs[-self.capacity:]
            actions = actions[-self.capacity:]
            rewards = rewards[-self.capacity:]
            next_obs = next_obs[-self.capacity:]
            terminals = terminals[-self.capacity:]
            n = self.capacity
        indices = (self.position + np.arange(n)) % self.capacity
        self.obs[indices] = obs
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_obs[indices] = next_obs
        self.terminals[indices] = terminals
        self.position = (self.position + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size):
        """
        :return: a dict of jax arrays {"obs", "actions", "rewards", "next_obs", "terminals"}
        """
        indices = self.np_random.choice(self.size, size=batch_size, replace=self.size < batch_size)
        return {
            "obs": jnp.asarray(self.obs[indices]),
            "actions": jnp.asarray(self.actions[indices]),
            "rewards": jnp.asarray(self.rewards[indices]),
            "next_obs": jnp.asarray(self.next_obs[indices]),
            "terminals": jnp.asarray(self.terminals[indices]),
        }

    def __len__(self):
        return self.size
