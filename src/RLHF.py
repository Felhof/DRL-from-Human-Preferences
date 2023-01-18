import multiprocessing

import gymnasium as gym


def start_collecting_trajectories():
    pass


def start_reward_modelling(reward_model_queue):
    print(reward_model_queue)


class RLHFWrapper:
    def __init__(self: "RLHFWrapper", environment: gym.Env) -> None:
        self.environment = environment
        self.current_observation = None
        self.reward_model = None
        self.reward_model_queue = "bla"
        self.trajectory_queue = None

    def start_rlhf(self: "RLHFWrapper") -> None:
        p1 = multiprocessing.Process(
            target=start_reward_modelling, args=self.reward_model_queue
        )

    def reset(self):
        pass

    def step(self: "RLHFWrapper", action):
        self.reward_model.get_reward(action, self.current_observation)
