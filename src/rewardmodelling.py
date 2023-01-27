from multiprocessing import Process
from typing import List

import numpy as np
from src.preferences import Preference, Queue
import torch
import torch.nn as nn

BUFFER_SIZE = 3000
EVALUATION_FREQ = 0.2
MIN_COMPARISONS_FOR_TRAINING = 500


class PreferenceBuffer:
    def __init__(self: "PreferenceBuffer", buffer_size=3000) -> None:
        self.preferences: List[Preference] = []
        self.number_of_preferences = 0
        self.idx = 0
        self.buffer_size = buffer_size

    def __len__(self: "PreferenceBuffer") -> int:
        return self.number_of_preferences

    def add(self, preference: Preference) -> None:
        if self.number_of_preferences < self.buffer_size:
            self.preferences.append(preference)
            self.number_of_preferences += 1
        else:
            self.preferences[self.idx] = preference
        self.idx = (self.idx + 1) % self.buffer_size

    def get_minibatch(self, n=32) -> List[Preference]:
        indices = list(range(0, self.number_of_preferences))
        minibatch_indices = np.random.choice(indices, size=n, replace=False)

        return [self.preferences[i] for i in minibatch_indices]


class RewardModel(torch.nn.Module):
    def __init__(self: "RewardModel") -> None:
        super().__init__()
        net = nn.Sequential()
        net.append(nn.Conv2d(in_channels=4, out_channels=16, kernel_size=7, stride=3))
        net.append(nn.Dropout2d(p=0.5))
        net.append(nn.BatchNorm2d(16))
        net.append(nn.LeakyReLU())
        net.append(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2))
        net.append(nn.Dropout2d(p=0.5))
        net.append(nn.BatchNorm2d(16))
        net.append(nn.LeakyReLU())
        net.append(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1))
        net.append(nn.Dropout2d(p=0.5))
        net.append(nn.BatchNorm2d(16))
        net.append(nn.LeakyReLU())
        net.append(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1))
        net.append(nn.Dropout2d(p=0.5))
        net.append(nn.BatchNorm2d(16))
        net.append(nn.LeakyReLU())
        net.append(nn.Flatten(start_dim=1))
        net.append(nn.Linear(in_features=784, out_features=64))
        net.append(nn.LeakyReLU())
        net.append(nn.Linear(in_features=64, out_features=1))
        self.net = net
        self.has_completed_pretraining = False

    def forward(self: "RewardModel", x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            output = self.net(x)
            return output.squeeze(0)
        return self.net(x)


class RewardModellingProcess(Process):
    def __init__(
        self: "RewardModellingProcess",
        preference_queue: Queue,
        reward_model_queue: Queue,
        stop_queue: Queue,
    ) -> None:
        super().__init__()
        self.preference_queue = preference_queue
        self.reward_model = RewardModel()
        self.reward_model_queue = reward_model_queue
        self.training_buffer = PreferenceBuffer(buffer_size=BUFFER_SIZE)
        self.evaluation_buffer = PreferenceBuffer(
            buffer_size=int(BUFFER_SIZE * EVALUATION_FREQ)
        )
        self.stop_queue = stop_queue

        self.reward_model_queue.put(self.reward_model)

    def run(self: "RewardModellingProcess") -> None:
        while True:
            if not self.stop_queue.empty():
                if self.stop_queue.get():
                    break

            while not self.preference_queue.empty():
                preference = self.preference_queue.get()
                use_for_training = np.random.binomial(1, p=1 - EVALUATION_FREQ)
                if use_for_training:
                    self.training_buffer.add(preference)
                else:
                    self.evaluation_buffer.add(preference)

            if (
                len(self.training_buffer) + len(self.evaluation_buffer)
                < MIN_COMPARISONS_FOR_TRAINING
            ):
                continue

            self.update_reward_model()

    def update_reward_model(self: "RewardModellingProcess") -> None:
        epochs = 1 if self.reward_model.has_completed_pretraining else 200

        for _ in range(epochs):
            self.train_reward_model_for_one_epoch()

        self.reward_model.has_completed_pretraining = True

        self.reward_model_queue.put(self.reward_model)

    def train_reward_model_for_one_epoch(self):
        pass
