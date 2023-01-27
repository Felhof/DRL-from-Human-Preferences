from multiprocessing import Process, Queue
import multiprocessing

import gymnasium as gym
from src.preferences import FeedbackCollectionProcess
from src.rewardmodelling import RewardModellingProcess


class RLHFWrapper:
    def __init__(self: "RLHFWrapper", environment: gym.Env) -> None:
        self.environment = environment
        self.current_observation = None
        self.reward_model = None
        self.reward_model_queue = Queue()
        self.stop_reward_modelling_queue = Queue()
        self.trajectory_queue = Queue()
        self.current_trajectory = []

    def start_rlhf(self: "RLHFWrapper") -> None:
        preference_queue = multiprocessing.Queue()
        feedback_collecting_process = FeedbackCollectionProcess(self.trajectory_queue)
        reward_modelling_process = RewardModellingProcess(
            preference_queue=preference_queue,
            reward_model_queue=self.reward_model_queue,
            stop_queue=self.stop_reward_modelling_queue,
        )
        feedback_collecting_process.start()
        reward_modelling_process.start()

    def reset(self):
        observation, info = self.environment.reset()
        self.current_observation = observation
        self.current_trajectory = []
        return observation, info

    def step(self: "RLHFWrapper", action):
        if not self.reward_model_queue.empty():
            self.reward_model = self.reward_model_queue.get()

        self.current_trajectory.append((self.current_observation, action))

        obs, _, terminated, truncated, info = self.environment.step(action)
        reward = self.reward_model.get_reward(self.current_observation, action)

        self.current_observation = obs

        if terminated or truncated:
            self.trajectory_queue.put(self.current_trajectory)
            self.current_trajectory = []

        return obs, reward, terminated, truncated, info
