from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.preferences import Preference

BUFFER_SIZE = 3000


class PreferenceBuffer:
    def __init__(self: "PreferenceBuffer") -> None:
        self.preferences: List[Preference] = []
        self.number_of_preferences = 0
        self.idx = 0

    def __len__(self: "PreferenceBuffer") -> int:
        return self.number_of_preferences

    def add(self, preference: Preference) -> None:
        if self.number_of_preferences < BUFFER_SIZE:
            self.preferences.append(preference)
            self.number_of_preferences += 1
        else:
            self.preferences[self.idx] = preference
        self.idx = (self.idx + 1) % BUFFER_SIZE

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

    def forward(self: "RewardModel", x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            output = self.net(x)
            return output.squeeze(0)
        return self.net(x)
