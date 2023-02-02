import logging
import logging.handlers
from multiprocessing import Process
import pickle
from time import sleep
from typing import List, Iterator

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

    def __eq__(self: "PreferenceBuffer", other: "PreferenceBuffer") -> bool:
        if len(self.preferences) != len(other.preferences):
            return False
        if self.number_of_preferences != other.number_of_preferences:
            return False
        if self.idx != other.idx:
            return False
        if self.buffer_size != other.buffer_size:
            return False
        return all([p1 == p2 for p1, p2 in zip(self.preferences, other.preferences)])

    def __len__(self: "PreferenceBuffer") -> int:
        return self.number_of_preferences

    def add(self, preference: Preference) -> None:
        if self.number_of_preferences < self.buffer_size:
            self.preferences.append(preference)
            self.number_of_preferences += 1
        else:
            self.preferences[self.idx] = preference
        self.idx = (self.idx + 1) % self.buffer_size

    def get_minibatches(self: "PreferenceBuffer", n=4) -> Iterator[List[Preference]]:
        indices = np.random.permutation(list(range(0, self.number_of_preferences)))

        batch_start_index = 0

        while batch_start_index + n < len(self) + 1:
            batch_indices = indices[batch_start_index: batch_start_index + n]
            minibatch = [self.preferences[i] for i in batch_indices]
            yield minibatch
            batch_start_index += n

    def save_to_file(
            self: "PreferenceBuffer", filename: str = "../data/preferences"
    ) -> None:
        with open(f"{filename}.ptk", "wb") as file:
            pickle.dump(self, file)

    def load_from_file(
            self: "PreferenceBuffer", filename: str = "../data/preferences"
    ) -> None:
        with open(f"{filename}.ptk", "rb") as file:
            loaded_buffer: PreferenceBuffer = pickle.load(file)
        self.preferences = loaded_buffer.preferences
        self.number_of_preferences = loaded_buffer.number_of_preferences
        self.idx = loaded_buffer.idx
        self.buffer_size = loaded_buffer.buffer_size


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
            preference_source: str = "",
            preference_target: str = "",
            save_buffers_every_n_preferences: int = 10,
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

        self.reward_model_optimizer = torch.optim.Adam(
            self.reward_model.parameters(), lr=0.0001
        )

        self.logger = logging.getLogger(self.name)

        self.preference_source = preference_source
        self.preference_target = preference_target
        self.save_buffers_every_n_preferences = save_buffers_every_n_preferences

    def _load_preference_buffers(self: "RewardModellingProcess") -> None:
        self.logger.info("Trying to load preferences from file.")
        try:
            self.training_buffer.load_from_file(self.preference_source + "_training")
            self.evaluation_buffer.load_from_file(
                self.preference_source + "_evaluation"
            )
        except FileNotFoundError as e:
            self.logger.info(f"Error when trying to load preferences: {str(e)}")
            return
        self.logger.info(
            f"Successfully loaded {len(self.training_buffer)} preferences for training "
            f"and {len(self.evaluation_buffer)} for evaluation."
        )

    def _save_preference_buffers(self: "RewardModellingProcess") -> None:
        self.logger.info("Trying to save preferences to file.")
        self.training_buffer.save_to_file(self.preference_target + "_training")
        self.evaluation_buffer.save_to_file(self.preference_target + "_evaluation")
        self.logger.info("Successfully saved collected preferences.")

    def run(
            self: "RewardModellingProcess",
    ) -> None:
        self.logger.info("Starting reward modelling process.")
        if self.preference_source == "":
            self.logger.info("No filepath for previously stored preferences was given.")
        else:
            self._load_preference_buffers()

        preference_count = 0
        self.logger.info("Starting reward modelling loop.")
        while True:
            if self.stop_queue.qsize() != 0:
                if self.stop_queue.get():
                    self.logger.info("Received stop signal.")
                    break

            if self.preference_queue.qsize() == 0:
                self.logger.info("No preferences available currently.")
                sleep(5)

            while self.preference_queue.qsize() != 0:
                preference_count += 1
                preference = self.preference_queue.get()
                use_for_training = np.random.binomial(1, p=1 - EVALUATION_FREQ)
                if use_for_training:
                    self.logger.info(
                        "Got preference and adding it to the training buffer."
                    )
                    self.training_buffer.add(preference)
                else:
                    self.logger.info(
                        "Got preference and adding it to the evaluation buffer."
                    )
                    self.evaluation_buffer.add(preference)

                if (
                        self.preference_target != ""
                        and preference_count % self.save_buffers_every_n_preferences == 0
                ):
                    self._save_preference_buffers()

            if (
                    len(self.training_buffer) + len(self.evaluation_buffer)
                    < MIN_COMPARISONS_FOR_TRAINING
            ):
                self.logger.info(
                    "Not enough preferences for training the reward model yet."
                )
                continue

            self.update_reward_model()

    def update_reward_model(self: "RewardModellingProcess") -> None:
        if self.reward_model.has_completed_pretraining:
            self.logger.info("Training model for one epoch.")
            epochs = 1
        else:
            self.logger.info("Model is not pretrained yet so training for 200 epochs.")
            epochs = 200

        for _ in range(epochs):
            self.train_reward_model_for_one_epoch()

        self.reward_model.has_completed_pretraining = True

        self.reward_model_queue.put(self.reward_model)

    def _get_loss_for_minibatch(
            self: "RewardModellingProcess", minibatch
    ) -> torch.Tensor():
        estimated_rewards = []
        preference_distribution = []

        for preference in minibatch:
            r1 = torch.sum(
                self.reward_model(
                    torch.tensor(
                        preference.segment1.get_observations(), dtype=torch.float32
                    )
                )
            )
            r2 = torch.sum(
                self.reward_model(
                    torch.tensor(
                        preference.segment2.get_observations(), dtype=torch.float32
                    )
                )
            )
            estimated_rewards.append(torch.stack((r1, r2)))
            preference_distribution.append([1.0 - preference.mu, preference.mu])

        loss = nn.CrossEntropyLoss()(
            input=torch.stack(estimated_rewards),
            target=torch.tensor(preference_distribution, dtype=torch.float32),
        )

        return loss

    def train_reward_model_for_one_epoch(self: "RewardModellingProcess") -> None:
        for minibatch in self.training_buffer.get_minibatches():
            loss = self._get_loss_for_minibatch(minibatch)

            self.logger.info(f"The training loss this epoch is {loss}.")

            self.reward_model_optimizer.zero_grad()
            loss.backward()
            self.reward_model_optimizer.step()

        self.logger.info("Finished training for one epoch. Evaluating.")
        self.evaluate_model()

    def evaluate_model(self: "RewardModellingProcess") -> None:
        self.reward_model.eval()
        batch_losses = []
        for minibatch in self.evaluation_buffer.get_minibatches():
            batch_loss = self._get_loss_for_minibatch(minibatch).item()
            batch_losses.append(batch_loss)
        loss = np.mean(batch_losses)
        self.logger.info(f"Finished evaluating model. Evaluation loss is: {loss}")
        self.reward_model.train()
