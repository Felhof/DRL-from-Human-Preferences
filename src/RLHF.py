from multiprocessing import Process, Queue
import multiprocessing

import gymnasium as gym


def start_collecting_feedback(trajectory_queue: Queue):
    pass


def start_reward_modelling(reward_model_queue: Queue):
    pass


class RLHFWrapper:
    def __init__(self: "RLHFWrapper", environment: gym.Env) -> None:
        self.environment = environment
        self.current_observation = None
        self.reward_model = None
        self.reward_model_queue = Queue()
        self.trajectory_queue = Queue()

    def start_rlhf(self: "RLHFWrapper") -> None:
        feedback_collecting_process = multiprocessing.Process(
            target=start_collecting_feedback, args=(self.trajectory_queue,)
        )
        reward_modelling_process = multiprocessing.Process(
            target=start_reward_modelling, args=(self.reward_model_queue,)
        )
        feedback_collecting_process.start()
        reward_modelling_process.start()

    def reset(self):
        observation, info = self.environment.reset()
        self.current_observation = observation
        return observation, info

    def step(self: "RLHFWrapper", action):
        if not self.reward_model_queue.empty():
            self.reward_model = self.reward_model_queue.get()

        obs, _, terminated, truncated, info = self.environment.step(action)
        reward = self.reward_model.get_reward(obs, action)

        self.current_observation = obs

        return obs, reward, terminated, truncated, info
