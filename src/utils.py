from collections import deque
from typing import List, Tuple

import gym
import numpy as np
from gym import spaces

Trajectory = List[Tuple[np.ndarray, np.ndarray]]


class FrameStack(gym.Wrapper):
    def __init__(self: "FrameStack", env: gym.Env, n_frames: int = 4) -> None:
        super().__init__(env)
        self.frames = deque(
            [np.zeros((1, 84, 84)) for _ in range(n_frames)],
            maxlen=n_frames,
        )
        low = np.repeat(env.observation_space.low.reshape((1, 84, 84)), 4, axis=0)
        high = np.repeat(env.observation_space.high.reshape((1, 84, 84)), 4, axis=0)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def reset(self: "FrameStack"):
        obs = self.env.reset()
        self.frames.append(obs.reshape((1, 84, 84)))
        return LazyFrames(self.frames)

    def step(self: "FrameStack", action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs.reshape((1, 84, 84)))
        return LazyFrames(self.frames), reward, done, info


class LazyFrames:
    """
    Basically a minimalist implementation of LazyFrames from OpenAI's Atari Wrapper:
    https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/atari_wrappers.py#L229
    """

    def __init__(self, frames: deque) -> None:
        self.frames = list(frames)

    def __array__(self, dtype=None):
        array = np.concatenate(self.frames, axis=0)
        if dtype is not None:
            array = array.astype(dtype)
        return array

    def __getitem__(self, i):
        return self.frames[i]
