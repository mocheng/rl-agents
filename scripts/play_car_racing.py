"""Play CarRacing manually with keyboard controls.

Usage:
    .venv/bin/python scripts/play_car_racing.py            # discrete actions (5 buttons)
    .venv/bin/python scripts/play_car_racing.py continuous # continuous steering/gas/brake
    .venv/bin/python scripts/play_car_racing.py --seed 42
"""

import argparse

import numpy as np

import gymnasium as gym
from gymnasium.utils.play import play


def make_env(continuous: bool, render_mode: str = "rgb_array"):
    # NOTE: play() renders the frames itself, so the env must use "rgb_array".
    return gym.make("CarRacing-v3", continuous=continuous, render_mode=render_mode)


def keys_to_action_discrete():
    return {
        ("w",): 3,             # accelerate
        ("s",): 4,             # brake
        ("a",): 2,             # steer left
        ("d",): 1,             # steer right
    }


def keys_to_action_continuous():
    return {
        ("w",): np.array([0.0, 1.0, 0.0], dtype=np.float32),            # gas
        ("s",): np.array([0.0, 0.0, 1.0], dtype=np.float32),            # brake
        ("a",): np.array([-1.0, 0.0, 0.0], dtype=np.float32),           # steer left
        ("d",): np.array([1.0, 0.0, 0.0], dtype=np.float32),            # steer right
        ("w", "a"): np.array([-1.0, 1.0, 0.0], dtype=np.float32),       # gas + left
        ("w", "d"): np.array([1.0, 1.0, 0.0], dtype=np.float32),        # gas + right
        ("s", "a"): np.array([-1.0, 0.0, 1.0], dtype=np.float32),       # brake + left
        ("s", "d"): np.array([1.0, 0.0, 1.0], dtype=np.float32),        # brake + right
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="discrete", choices=["discrete", "continuous"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None, help="target frame rate (default: env FPS)")
    parser.add_argument("--zoom", type=float, default=3.0)
    args = parser.parse_args()

    continuous = args.mode == "continuous"
    env = make_env(continuous=continuous)
    keys_to_action = keys_to_action_continuous() if continuous else keys_to_action_discrete()

    noop = np.zeros(3, dtype=np.float32) if continuous else 0
    play(
        env,
        keys_to_action=keys_to_action,
        noop=noop,  # applied when no mapped key combination is pressed
        fps=args.fps,
        zoom=args.zoom,
        seed=args.seed,
    )
    env.close()


if __name__ == "__main__":
    main()
