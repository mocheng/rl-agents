import gymnasium as gym
import numpy as np
from collections import deque
from gymnasium import spaces


class Float64ActionWrapper(gym.ActionWrapper):
    """
        Cast actions to float64.

        Workaround for Box2D environments (e.g. CarRacing) whose SWIG bindings reject float32
        actions with "argument 2 of type 'float32'" when `box2d-py` is built against NumPy 2.
    """

    def action(self, action):
        return np.asarray(action, dtype=np.float64)


class PunishActionReward(gym.RewardWrapper):
    """
        Subtract a penalty whenever a given action is chosen.

        Used for CarRacing, where doing nothing (action 0) is a local optimum: the car collects the
        per-step time penalty while avoiding the risk of driving off the track.
    """

    def __init__(self, env, action=0, penalty=0.1):
        super().__init__(env)
        self.action = action
        self.penalty = penalty
        self.last_action = None

    def reset(self, **kwargs):
        self.last_action = None
        return super().reset(**kwargs)

    def step(self, action):
        self.last_action = action
        return super().step(action)

    def reward(self, reward):
        if self.last_action is not None and int(self.last_action) == self.action:
            reward = reward - self.penalty
        return reward


class PunishStallReward(gym.RewardWrapper):
    """
        Penalize stalling: subtract a penalty whenever the vehicle's speed is below a threshold.

        Unlike punishing a fixed action, this targets the stalling behaviour itself and leaves
        coasting at speed unpenalized. The penalty can optionally be restricted to a given action.
    """

    def __init__(self, env, min_speed=2.0, penalty=0.1, action=None, speed_fn=None):
        super().__init__(env)
        self.min_speed = min_speed
        self.penalty = penalty
        self.action = action
        self.speed_fn = speed_fn or self._car_speed
        self.last_action = None
        self.last_speed = None

    def reset(self, **kwargs):
        self.last_action = None
        self.last_speed = None
        return super().reset(**kwargs)

    def step(self, action):
        self.last_action = action
        self.last_speed = self.speed_fn(self.env)
        return super().step(action)

    def reward(self, reward):
        if self.last_speed is not None and self.last_speed < self.min_speed:
            if self.action is None or int(self.last_action) == self.action:
                reward = reward - self.penalty
        return reward

    @staticmethod
    def _car_speed(env):
        try:
            return float(env.unwrapped.car.hull.linearVelocity.length)
        except AttributeError:
            return float("inf")


class FrameSkip(gym.Wrapper):
    """
        Repeat each action for `skip` environment steps and sum the rewards.

        Standard practice for pixel-based control: it reduces the decision frequency, lengthens the
        effective planning horizon and stabilizes the learned control.
    """

    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        observation, info = None, {}
        for _ in range(self.skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return observation, total_reward, terminated, truncated, info


class PreprocessImage(gym.ObservationWrapper):
    """
        Preprocess image observations: optional grayscale, downscaling and frame stacking.

        This is the standard recipe for pixel-based RL (e.g. CarRacing: grayscale, 84x84, 4 stacked
        frames). The observation is returned in channel-first layout `(frame_stack, size, size)` for
        grayscale, so that convolutional policies (e.g. Stable-Baselines3's CnnPolicy) treat the
        stacked frames as image channels.
    """

    def __init__(self, env, size=84, grayscale=True, frame_stack=1):
        super().__init__(env)
        self.size = size
        self.grayscale = grayscale
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        channels = 1 if grayscale else env.observation_space.shape[-1]
        if frame_stack > 1 and channels != 1:
            raise ValueError("Frame stacking is only supported for grayscale observations.")
        self.observation_space = spaces.Box(0, 255, (frame_stack, size, size, channels), dtype=np.uint8)
        if channels == 1:
            self.observation_space = spaces.Box(0, 255, (frame_stack, size, size), dtype=np.uint8)

    def reset(self, **kwargs):
        self.frames.clear()
        return super().reset(**kwargs)

    def observation(self, observation):
        frame = self._preprocess(observation)
        while len(self.frames) < self.frame_stack:
            self.frames.append(frame)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

    def _preprocess(self, observation):
        frame = np.asarray(observation)
        if self.grayscale:
            frame = np.dot(frame[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
        else:
            frame = frame.astype(np.float32)
        return self._resize(frame.astype(np.uint8), self.size)

    @staticmethod
    def _resize(frame, size):
        squeeze = frame.ndim == 2
        if squeeze:
            frame = frame[..., None]
        try:
            from PIL import Image
            image = Image.fromarray(frame.squeeze(-1) if frame.shape[-1] == 1 else frame)
            resized = np.asarray(image.resize((size, size), Image.BILINEAR))
            resized = resized[..., None] if frame.shape[-1] == 1 else resized
        except ImportError:
            # Nearest-neighbour fallback without Pillow
            height, width = frame.shape[:2]
            rows = np.linspace(0, height - 1, size).astype(int)
            columns = np.linspace(0, width - 1, size).astype(int)
            resized = frame[np.ix_(rows, columns, np.arange(frame.shape[2]))]
        return resized.squeeze(-1) if squeeze else resized
