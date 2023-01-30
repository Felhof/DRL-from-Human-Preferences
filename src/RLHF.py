from multiprocessing import Queue
import multiprocessing
from typing import Any

import gym
import numpy as np
import torch

from src.preferences import FeedbackCollectionProcess
from src.rewardmodelling import RewardModel, RewardModellingProcess


class RLHFWrapper(gym.Wrapper):
    def __init__(self: "RLHFWrapper", environment: gym.Env) -> None:
        super().__init__(environment)
        self.environment = environment
        self.current_observation = None
        self.reward_model: RewardModel = None
        self.reward_model_queue = Queue()
        self.stop_reward_modelling_queue = Queue()
        self.trajectory_queue = Queue()
        self.stop_feedback_collecting_queue = Queue()
        self.current_trajectory = []
        self.feedback_collecting_process = None
        self.reward_modelling_process = None

    def start_rlhf(self: "RLHFWrapper") -> None:
        preference_queue = multiprocessing.Queue()
        self.feedback_collecting_process = FeedbackCollectionProcess(
            preference_queue=preference_queue,
            trajectory_queue=self.trajectory_queue,
            stop_queue=self.stop_feedback_collecting_queue,
        )
        self.reward_modelling_process = RewardModellingProcess(
            preference_queue=preference_queue,
            reward_model_queue=self.reward_model_queue,
            stop_queue=self.stop_reward_modelling_queue,
        )
        self.feedback_collecting_process.start()
        self.reward_modelling_process.start()

        while True:
            if not self.reward_model_queue.empty():
                self.reward_model = self.reward_model_queue.get()
                break

    def render(self, mode="human") -> Any:
        return self.environment.render(mode=mode)

    def reset(self):
        observation = self.environment.reset()
        self.current_observation = np.array(observation)
        self.current_trajectory = []
        return observation

    def step(self: "RLHFWrapper", action):
        if not self.reward_model_queue.empty():
            self.reward_model = self.reward_model_queue.get()

        self.current_trajectory.append((self.current_observation, action))

        obs, _, done, info = self.environment.step(action)
        reward = self.reward_model(torch.tensor(np.array(obs), dtype=torch.float32))

        self.current_observation = np.array(obs)

        if done:
            self.trajectory_queue.put(self.current_trajectory)
            self.current_trajectory = []

        return obs, reward.item(), done, info

    def stop(self: "RLHFWrapper") -> None:
        self.stop_feedback_collecting_queue.put(True)
        self.stop_reward_modelling_queue.put(True)
        self.feedback_collecting_process.terminate()
        self.reward_modelling_process.terminate()
