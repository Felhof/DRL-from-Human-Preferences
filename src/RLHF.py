import logging
import logging.handlers
from multiprocessing import Queue
from typing import Any

import gym
import numpy as np
import torch

from src.loglistening import LogListener
from src.preferences import FeedbackCollectionProcess
from src.rewardmodelling import RewardModel, RewardModellingProcess

TRAJECTORY_QUEUE_CAPACITY = 5


class RLHFWrapper(gym.Wrapper):
    def __init__(self: "RLHFWrapper", environment: gym.Env) -> None:
        super().__init__(environment)
        self.environment = environment
        self.current_observation = None
        self.reward_model: RewardModel = None
        self.reward_model_queue = Queue()
        self.stop_reward_modelling_queue = Queue()
        self.trajectory_queue = Queue(TRAJECTORY_QUEUE_CAPACITY)
        self.stop_feedback_collecting_queue = Queue()
        self.current_trajectory = []
        self.feedback_collecting_process = None
        self.reward_modelling_process = None
        self.logger = None
        self.log_listener = None

    def start_rlhf(
            self: "RLHFWrapper",
            preference_source: str = "",
            preference_target: str = "",
    ) -> None:
        log_queue = Queue()

        self.log_listener = LogListener(queue=log_queue)
        self.log_listener.start()

        log_queue_handler = logging.handlers.QueueHandler(log_queue)
        root_logger = logging.getLogger()
        root_logger.addHandler(log_queue_handler)
        self.logger = logging.getLogger("main")

        preference_queue = Queue()
        self.feedback_collecting_process = FeedbackCollectionProcess(
            preference_queue=preference_queue,
            trajectory_queue=self.trajectory_queue,
            stop_queue=self.stop_feedback_collecting_queue,
        )
        self.reward_modelling_process = RewardModellingProcess(
            preference_queue=preference_queue,
            reward_model_queue=self.reward_model_queue,
            stop_queue=self.stop_reward_modelling_queue,
            preference_source=preference_source,
            preference_target=preference_target
        )
        self.feedback_collecting_process.start()
        self.reward_modelling_process.start()

        while True:
            if not self.reward_model_queue.empty():
                self.logger.info("Got initialized reward model.")
                self.reward_model = self.reward_model_queue.get()
                break

    def render(self, mode="human") -> Any:
        return self.environment.render(mode=mode)

    def reset(self):
        self.logger.info("Resetting environment.")
        observation = self.environment.reset()
        self.current_observation = np.array(observation)
        self.current_trajectory = []
        return observation

    def step(self: "RLHFWrapper", action):
        if not self.reward_model_queue.empty():
            self.logger.info("Received new reward model from the queue.")
            self.reward_model = self.reward_model_queue.get()

        self.current_trajectory.append((self.current_observation, action))

        obs, _, done, info = self.environment.step(action)
        reward = self.reward_model(torch.tensor(np.array(obs), dtype=torch.float32))

        self.current_observation = np.array(obs)

        if done:
            self.logger.info("Episode ended.")
            if not self.trajectory_queue.full():
                self.logger.info(
                    "Putting the current trajectory in the trajectory queue."
                )
                self.trajectory_queue.put(self.current_trajectory)
            else:
                self.logger.info(
                    "Trajectory queue is full. Dropping current trajectory."
                )
            self.current_trajectory = []

        return obs, reward.item(), done, info

    def stop(self: "RLHFWrapper") -> None:
        self.logger.info("Stopping RLHF.")
        self.stop_feedback_collecting_queue.put(True)
        self.stop_reward_modelling_queue.put(True)
        self.feedback_collecting_process.terminate()
        self.reward_modelling_process.terminate()
        self.log_listener.terminate()
