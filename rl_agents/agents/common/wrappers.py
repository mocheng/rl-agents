import gymnasium as gym
import numpy as np


class Float64ActionWrapper(gym.ActionWrapper):
    """
        Cast actions to float64.

        Workaround for Box2D environments (e.g. CarRacing) whose SWIG bindings reject float32
        actions with "argument 2 of type 'float32'" when `box2d-py` is built against NumPy 2.
    """

    def action(self, action):
        return np.asarray(action, dtype=np.float64)
