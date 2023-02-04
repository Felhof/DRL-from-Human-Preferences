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

    def get_minibatches(self: "PreferenceBuffer", n=32) -> Iterator[List[Preference]]:
        indices = np.random.permutation(list(range(0, self.number_of_preferences)))

        batch_start_index = 0

        while batch_start_index + n < len(self) + 1:
            batch_indices = indices[batch_start_index : batch_start_index + n]
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
        collected_initial_preferences_queue: Queue,
        mode: str = "full_rlhf",
        preference_source: str = "",
        preference_target: str = "",
        save_buffers_every_n_preferences: int = 50,
    ) -> None:
        super().__init__()
        self.preference_queue = preference_queue
        self.reward_model = RewardModel()
        self.reward_model_queue = reward_model_queue
        self.collected_initial_preferences_queue = collected_initial_preferences_queue
        self.training_buffer = PreferenceBuffer(buffer_size=BUFFER_SIZE)
        self.evaluation_buffer = PreferenceBuffer(
            buffer_size=int(BUFFER_SIZE * EVALUATION_FREQ)
        )
        self.stop_queue = stop_queue

        self.reward_model_optimizer = torch.optim.Adam(
            self.reward_model.parameters(), lr=0.0001
        )

        self.logger = logging.getLogger(self.name)

        self.preference_source = preference_source
        self.preference_target = preference_target
        self.save_buffers_every_n_preferences = save_buffers_every_n_preferences
        self.mode = mode

    def _collect_initial_preferences(self: "RewardModellingProcess", n=500) -> None:
        while self._number_of_stored_preferences() < n:
            received_new_preference = (
                self._try_to_store_preference_from_queue_in_buffer()
            )
            if not received_new_preference:
                sleep(1)
                continue
            if (
                self.preference_target != ""
                and self._number_of_stored_preferences()
                % self.save_buffers_every_n_preferences
                == 0
            ):
                self._save_preference_buffers()

        self.logger.info(f"Finished collecting initial {n} preferences.")
        self._save_preference_buffers()
        self.collected_initial_preferences_queue.put(True)

    def _evaluate_model(self: "RewardModellingProcess") -> float:
        self.logger.info("Evaluating reward model.")
        print("Evaluating reward model.")
        self.reward_model.eval()
        batch_losses = []
        for minibatch in self.evaluation_buffer.get_minibatches():
            batch_loss = self._get_loss_for_minibatch(minibatch)
            batch_losses.append(batch_loss.detach().numpy())
        loss = np.mean(batch_losses).item()
        self.logger.info(f"Evaluation loss: {loss}")
        print(f"Evaluation loss: {loss}")
        self.reward_model.train()
        return loss

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

    def _load_preference_buffers(self: "RewardModellingProcess") -> None:
        self.logger.info("Trying to load preferences from file.")
        try:
            self.training_buffer.load_from_file(self.preference_source + "_training")
            self.evaluation_buffer.load_from_file(
                self.preference_source + "_evaluation"
            )
        except FileNotFoundError as e:
            self.logger.info(f"Error when trying to load preferences: {str(e)}")
            print(f"Error when trying to load preferences: {str(e)}")
            return
        self.logger.info(
            f"Successfully loaded {len(self.training_buffer)} preferences for training "
            f"and {len(self.evaluation_buffer)} for evaluation."
        )

    def _number_of_stored_preferences(self: "RewardModellingProcess") -> int:
        return len(self.training_buffer) + len(self.evaluation_buffer)

    def _pretrain_reward_model(
        self: "RewardModellingProcess", n_pretraining_epochs=5
    ) -> None:
        for epoch in range(n_pretraining_epochs):
            self.logger.info(
                f"Beginning training epoch {epoch + 1} of {n_pretraining_epochs}."
            )
            print(f"Beginning training epoch {epoch + 1} of {n_pretraining_epochs}.")
            self._train_reward_model_for_one_epoch()
            self.logger.info(
                f"Finished training epoch {epoch + 1} of {n_pretraining_epochs}."
            )
            print(f"Finished training epoch {epoch + 1} of {n_pretraining_epochs}.")
            self._evaluate_model()

        self.logger.info(f"Putting reward model in the reward model queue.")
        self.reward_model_queue.put(self.reward_model)

    def _reward_model_training_loop(self: "RewardModellingProcess") -> None:
        received_preferences_since_last_update = 0
        self.logger.info("Starting reward model training loop.")
        while True:
            if self.stop_queue.qsize() != 0:
                if self.stop_queue.get():
                    self.logger.info("Received stop signal.")
                    break

            received_new_preference = (
                self._try_to_store_preference_from_queue_in_buffer()
            )

            if received_new_preference:
                received_preferences_since_last_update += 1
            else:
                sleep(1)
                continue

            self.logger.info(
                f"A total number of {self._number_of_stored_preferences()} preferences are available."
            )

            if (
                self.preference_target != ""
                and self._number_of_stored_preferences()
                % self.save_buffers_every_n_preferences
                == 0
            ):
                self._save_preference_buffers()

            if received_preferences_since_last_update < 32:
                self.logger.info(
                    "Not updating model since not enough preferences were received since the last update."
                )
                continue

            self.logger.info("Updating reward model by training for one epoch.")
            received_preferences_since_last_update = 0
            self._train_reward_model_for_one_epoch()
            self._evaluate_model()
            self.reward_model_queue.put(self.reward_model)

    def _save_preference_buffers(self: "RewardModellingProcess") -> None:
        self.logger.info("Trying to save preferences to file.")
        self.training_buffer.save_to_file(self.preference_target + "_training")
        self.evaluation_buffer.save_to_file(self.preference_target + "_evaluation")
        self.logger.info("Successfully saved collected preferences.")

    def _train_reward_model_for_one_epoch(self: "RewardModellingProcess") -> None:
        batch_losses = []
        for minibatch in self.training_buffer.get_minibatches():
            loss = self._get_loss_for_minibatch(minibatch)

            self.reward_model_optimizer.zero_grad()
            loss.backward()
            self.reward_model_optimizer.step()
            batch_losses.append(loss.detach().numpy())

        self.logger.info(
            f"The mean training loss this epoch was {np.mean(batch_losses)}."
        )
        print(f"The mean training loss this epoch was {np.mean(batch_losses)}.")

    def _try_to_store_preference_from_queue_in_buffer(
        self: "RewardModellingProcess",
    ) -> bool:
        self.logger.info("Trying to get a new preference from the preference queue.")
        if self.preference_queue.qsize() != 0:
            preference = self.preference_queue.get()
            use_for_training = np.random.binomial(1, p=1 - EVALUATION_FREQ)
            if use_for_training:
                self.logger.info("Got preference and adding it to the training buffer.")
                self.training_buffer.add(preference)
            else:
                self.logger.info(
                    "Got preference and adding it to the evaluation buffer."
                )
                self.evaluation_buffer.add(preference)
            return True
        else:
            self.logger.info("No new preferences available currently.")
            return False

    def run(
        self: "RewardModellingProcess",
    ) -> None:
        self.logger.info("Starting reward modelling process.")

        if self.mode == "full_rlhf" or self.mode == "collect_initial_preferences":
            self._collect_initial_preferences()

        if self.mode == "full_rlhf" or self.mode == "start_with_pretraining":
            if self.preference_source == "":
                self.logger.info(
                    "No filepath for previously stored preferences was given."
                )
                return
            self._load_preference_buffers()
            self._pretrain_reward_model()

        if self.mode == "full_rlhf" or self.mode == "start_with_pretraining":
            self._reward_model_training_loop()
