import logging
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jumanji.types import StepType

from jax_agents.agents.base import JaxAgent
from jax_agents.agents.memory import ReplayBuffer
from jax_agents.agents.networks import network_factory
from jax_agents.envs.jumanji import VectorizedJumanjiEnv

logger = logging.getLogger(__name__)


class JaxDQNAgent(JaxAgent):
    """
        A Deep Q-Network trained with jitted, vectorized rollouts on a Jumanji environment.

        The agent follows the repository's agent interface: `act` supports the usual
        single-step evaluation loop (`Evaluation.run_episodes`, `experiments.py --test`),
        while training is performed by `train_jax(evaluation)`, called by
        `jax_agents.trainer.JaxEvaluation`: a `num_envs`-parallel rollout is collected with
        `jax.lax.scan` under `jax.jit(jax.vmap(...))`, stored in a replay buffer, and
        followed by jitted double-DQN gradient steps with a target network.

        Configuration keys mirror the repository's pytorch DQN where meaningful
        (gamma, batch_size, memory_capacity, target_update, double, exploration), plus
        vectorization and JAX-specific keys (num_envs, rollout_steps, updates_per_batch,
        total_steps, learning_rate, grad_norm_clipping, device).
    """

    def __init__(self, env, config=None):
        super().__init__(env, config)
        unwrapped = getattr(env, "unwrapped", env)
        jumanji_env = getattr(unwrapped, "jumanji_env", None)
        if jumanji_env is None:
            raise ValueError("JaxDQNAgent requires an environment wrapped by "
                             "jax_agents.envs.jumanji.JumanjiEnv (id 'jumanji/...').")
        self.jumanji_env = jumanji_env
        self.vec_env = VectorizedJumanjiEnv(jumanji_env)
        self.num_actions = int(jumanji_env.action_spec.num_values)
        self.obs_shape = tuple(jumanji_env.observation_spec.grid.shape)
        self.network = network_factory(self.config["model"], num_actions=self.num_actions)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.config["grad_norm_clipping"]),
            optax.adam(self.config["learning_rate"]))
        self.params = None
        self.target_params = None
        self.opt_state = None
        self.buffer = None
        self.steps = 0  # Number of gradient steps (target network schedule)
        self._forward = jax.jit(lambda params, obs: self.network.apply({"params": params}, obs))
        self._rollout = None
        self._train_step = None

    @classmethod
    def default_config(cls):
        return dict(model=dict(type="ConvEncoder", channels=[32, 64], head_layers=[128]),
                    gamma=0.99,
                    learning_rate=3e-4,
                    grad_norm_clipping=10.0,
                    batch_size=256,
                    memory_capacity=100000,
                    target_update=500,
                    double=True,
                    num_envs=64,
                    rollout_steps=16,
                    updates_per_batch=4,
                    total_steps=200000,
                    n_steps=1,
                    exploration=dict(method="EpsilonGreedy", initial=1.0, final=0.01, tau=50000),
                    device="auto")

    def reset(self):
        """
            Initialize the value network, optimizer and replay buffer from the seeded key.
            The first reset initializes the model: it is called by the trainer before
            training, and by the evaluation loop before testing. If a model has been
            loaded (or parameters already initialized), this is a no-op.
        """
        if self._initialized:
            return
        self.key, init_key = jax.random.split(self.key)
        dummy_obs = jnp.zeros((1,) + self.obs_shape, jnp.float32)
        self.params = jax.device_put(self.network.init(init_key, dummy_obs)["params"], self.device)
        self.target_params = self.params
        self.opt_state = self.optimizer.init(self.params)
        self.buffer = ReplayBuffer(self.config["memory_capacity"], self.obs_shape,
                                   n_steps=self.config["n_steps"], gamma=self.config["gamma"])
        self.steps = 0
        self._initialized = True

    def act(self, state):
        """
            Pick a greedy action for a single state, for evaluation.
        """
        if not self._initialized:
            self.reset()
        obs = jnp.asarray(state, jnp.float32)[None]
        q = self._forward(self.params, obs)
        return int(np.asarray(jnp.argmax(q, axis=-1))[0])

    def _epsilon(self, step):
        """
            Linear epsilon schedule over environment steps.
        """
        exploration = self.config["exploration"]
        initial = float(exploration.get("initial", 1.0))
        final = float(exploration.get("final", 0.01))
        tau = float(exploration.get("tau", 50000))
        return final + (initial - final) * max(0.0, 1.0 - step / tau)

    def _make_rollout(self):
        """
            Build the jitted rollout function: a scan over `rollout_steps` of
            epsilon-greedy steps on `num_envs` parallel environments, with automatic
            reset of finished episodes.
        """
        num_envs = int(self.config["num_envs"])
        rollout_steps = int(self.config["rollout_steps"])
        num_actions = self.num_actions
        forward = self._forward
        vec_env = self.vec_env

        def broadcast(done, leaf):
            return jnp.reshape(done, (-1,) + (1,) * (leaf.ndim - 1))

        split_keys = jax.vmap(lambda key: jax.random.split(key, 4))
        random_action = jax.vmap(lambda key: jax.random.randint(key, (), 0, num_actions))
        random_probe = jax.vmap(lambda key: jax.random.uniform(key, ()))

        def rollout(params, states, keys, obs, returns, eps):
            def body(carry, _):
                states, keys, obs, returns = carry
                keys, a_keys, r_keys, reset_keys = [split_keys(keys)[:, i] for i in range(4)]
                q = forward(params, obs)                                # (num_envs, num_actions)
                greedy = jnp.argmax(q, axis=-1)
                random = random_action(a_keys)
                actions = jnp.where(random_probe(r_keys) < eps, random, greedy)
                next_states, timesteps = vec_env.step(states, actions)
                next_obs = vec_env.observation(timesteps.observation)    # (num_envs, rows, cols, channels)
                rewards = timesteps.reward                              # (num_envs,)
                done = timesteps.step_type == StepType.LAST             # (num_envs,)
                episode_returns = (returns + rewards) * done
                returns = jnp.where(done, jnp.zeros_like(returns), returns + rewards)
                reset_states, reset_timesteps = vec_env.reset(reset_keys)
                reset_obs = vec_env.observation(reset_timesteps.observation)
                states = jax.tree_util.tree_map(
                    lambda reset_leaf, next_leaf: jnp.where(broadcast(done, reset_leaf), reset_leaf, next_leaf),
                    reset_states, next_states)
                obs = jnp.where(broadcast(done, next_obs), reset_obs, next_obs)
                return (states, keys, obs, returns), \
                       (obs, actions, rewards, next_obs, done, episode_returns)

            carry, out = jax.lax.scan(body, (states, keys, obs, returns), None, length=rollout_steps)
            return carry, out

        self._rollout = jax.jit(rollout)

    def _make_train_step(self):
        """
            Build the jitted DQN gradient step (double Q-learning by default).
        """
        gamma = float(self.config["gamma"])
        double = bool(self.config["double"])
        forward = self._forward
        optimizer = self.optimizer

        def train_step(params, target_params, opt_state, batch):
            obs = batch["obs"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            next_obs = batch["next_obs"]
            terminals = batch["terminals"]
            batch_size = obs.shape[0]

            def loss_fn(params):
                q = forward(params, obs)                                # (batch_size, num_actions)
                q_values = q[jnp.arange(batch_size), actions]
                if double:
                    q_next = forward(params, next_obs)
                    best_actions = jnp.argmax(q_next, axis=-1)
                    next_values = forward(target_params, next_obs)[jnp.arange(batch_size), best_actions]
                else:
                    next_values = jnp.max(forward(target_params, next_obs), axis=-1)
                target = rewards + gamma * next_values * (1.0 - terminals)
                return jnp.mean((q_values - jax.lax.stop_gradient(target)) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return loss, params, opt_state

        self._train_step = jax.jit(train_step)

    def train_jax(self, evaluation):
        """
            Train the agent: collect vectorized rollouts, update the Q-network, and report
            episode statistics through the evaluation object (TensorBoard, checkpoints).
        """
        if not self._initialized:
            self.reset()
        self.training = True
        self._make_rollout()
        self._make_train_step()
        num_envs = int(self.config["num_envs"])
        rollout_steps = int(self.config["rollout_steps"])
        samples_per_epoch = num_envs * rollout_steps
        num_epochs = int(math.ceil(self.config["total_steps"] / samples_per_epoch))

        key = self.key
        key, reset_key = jax.random.split(key)
        keys = jax.random.split(reset_key, num_envs)
        states, timesteps = self.vec_env.reset(keys)
        obs = self.vec_env.observation(timesteps.observation)
        returns = jnp.zeros(num_envs, jnp.float32)
        self.set_global_step(evaluation.global_step)

        for epoch in range(num_epochs):
            start = time.time()
            eps = self._epsilon(evaluation.global_step)
            (states, keys, obs, returns), out = self._rollout(self.params, states, keys, obs, returns, eps)
            rollout_obs, actions, rewards, next_obs, terminals, episode_returns = out
            self.buffer.push(np.asarray(rollout_obs), np.asarray(actions), np.asarray(rewards),
                             np.asarray(next_obs), np.asarray(terminals).astype(np.float32))

            losses = []
            for _ in range(self.config["updates_per_batch"]):
                if len(self.buffer) < self.config["batch_size"]:
                    break
                batch = self.buffer.sample(self.config["batch_size"])
                loss, self.params, self.opt_state = \
                    self._train_step(self.params, self.target_params, self.opt_state, batch)
                self.steps += 1
                if self.steps % self.config["target_update"] == 0:
                    self.target_params = self.params
                losses.append(float(loss))

            evaluation.global_step += samples_per_epoch
            self.set_global_step(evaluation.global_step)
            duration = time.time() - start
            completed = np.asarray(episode_returns).reshape(-1)
            episode_rewards = completed[completed != 0].tolist()

            if self.writer is not None and losses:
                self.writer.add_scalar("agent/loss", float(np.mean(losses)), self.global_step)
                self.writer.add_scalar("agent/epsilon", eps, self.global_step)
                self.writer.add_scalar("agent/updates", self.steps, self.global_step)
                self.writer.add_scalar("agent/buffer_size", len(self.buffer), self.global_step)
            if episode_rewards:
                evaluation.after_all_episodes(epoch, episode_rewards, duration)
                evaluation.after_some_episodes(epoch, episode_rewards)
            logger.info("[JAX DQN epoch {}/{}] loss={:.4f} eps={:.3f} episodes={} env_steps={}".format(
                epoch + 1, num_epochs, np.mean(losses) if losses else float("nan"), eps,
                len(episode_rewards), evaluation.global_step))

    def _get_state(self):
        return ({"params": self.params,
                 "target_params": self.target_params,
                 "opt_state": self.opt_state},
                {"steps": self.steps, "global_step": self.global_step})

    def _set_state(self, state, extras):
        self.params = state["params"]
        self.target_params = state["target_params"]
        self.opt_state = state["opt_state"]
        self.steps = int(extras.get("steps", 0))
        self.global_step = int(extras.get("global_step", 0))
