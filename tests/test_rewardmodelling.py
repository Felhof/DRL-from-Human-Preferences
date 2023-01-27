import numpy as np
import pytest
import torch

import src.rewardmodelling
from src.preferences import Preference
from src.rewardmodelling import PreferenceBuffer, RewardModel, RewardModellingProcess


@pytest.fixture
def reward_modelling_process(mocker):
    def _create_reward_modelling_process(
        preference_queue=None,
        reward_model=None,
        reward_model_queue=None,
        stop_queue=None,
        training_buffer=None,
        evaluation_buffer=None,
        reward_model_optimizer=None,
    ):
        if preference_queue is None:
            preference_queue = mocker.Mock()

        if reward_model_queue is None:
            reward_model_queue = mocker.Mock()
            reward_model_queue.put = mocker.Mock()

        if stop_queue is None:
            stop_queue = mocker.Mock()

        if training_buffer is None:
            training_buffer = mocker.Mock()
            training_buffer.__len__ = mocker.Mock(return_value=1)

        if evaluation_buffer is None:
            evaluation_buffer = mocker.Mock()
            evaluation_buffer.__len__ = mocker.Mock(return_value=1)

        if reward_model_optimizer is None:
            reward_model_optimizer = mocker.Mock()

        mocker.patch(
            "src.rewardmodelling.RewardModel",
            return_value=reward_model if reward_model is not None else mocker.Mock(),
        )

        mocker.patch(
            "src.rewardmodelling.PreferenceBuffer",
            side_effect=[
                training_buffer,
                evaluation_buffer,
            ],
        )

        mocker.patch(
            "src.rewardmodelling.torch.optim.Adam", return_value=reward_model_optimizer
        )

        reward_modeller = RewardModellingProcess(
            preference_queue, reward_model_queue, stop_queue
        )

        return reward_modeller

    return _create_reward_modelling_process


@pytest.fixture
def runnable_reward_modelling_process(mocker, reward_modelling_process):
    def _create(use_for_training=None, **kwargs):
        if use_for_training is None:
            use_for_training = [1]
        mocker.patch(
            "src.rewardmodelling.np.random.binomial", side_effect=use_for_training
        )

        if "preference_queue" not in kwargs:
            preference_queue = mocker.Mock()
            preference_queue.empty = mocker.Mock(return_value=True)
            kwargs["preference_queue"] = preference_queue

        if "stop_queue" not in kwargs:
            stop_queue = mocker.Mock()
            stop_queue.empty = mocker.Mock(return_value=False)
            stop_queue.get = mocker.Mock(side_effect=[False, True])
            kwargs["stop_queue"] = stop_queue

        reward_modeller = reward_modelling_process(**kwargs)
        return reward_modeller

    return _create


def test_preference_buffer_can_add_new_items_up_to_buffer_size_and_loops_afterwards(
    mocker,
):
    # Given
    p1 = mocker.Mock()
    p2 = mocker.Mock()
    p3 = mocker.Mock()
    p4 = mocker.Mock()

    # When
    preference_buffer = PreferenceBuffer(3)

    for p in [p1, p2, p3]:
        preference_buffer.add(p)

    # Then
    assert len(preference_buffer) == 3
    assert preference_buffer.preferences[0] == p1
    assert preference_buffer.preferences[1] == p2
    assert preference_buffer.preferences[2] == p3

    # When
    preference_buffer.add(p4)

    # Then
    assert len(preference_buffer) == 3
    assert preference_buffer.preferences[0] == p4
    assert preference_buffer.preferences[1] == p2
    assert preference_buffer.preferences[2] == p3


def test_preference_buffer_can_get_minibatches(mocker):
    # Given
    p1 = mocker.Mock()
    p2 = mocker.Mock()
    p3 = mocker.Mock()
    p4 = mocker.Mock()
    p5 = mocker.Mock()
    p6 = mocker.Mock()
    mocker.patch(
        "src.rewardmodelling.np.random.permutation",
        side_effect=[np.array([1, 4, 2, 5, 0, 3])],
    )

    # When
    preference_buffer = PreferenceBuffer()

    for p in [p1, p2, p3, p4, p5, p6]:
        preference_buffer.add(p)

    minibatches = [minibatch for minibatch in preference_buffer.get_minibatches(n=3)]

    # Then
    assert minibatches == [[p2, p5, p3], [p6, p1, p4]]


def test_reward_model_maps_observation_to_scalar():
    reward_model = RewardModel()

    single_frame = torch.rand((4, 84, 84))
    prediction = reward_model.forward(single_frame)

    assert prediction.shape == torch.Size([1])

    minibatch = torch.rand((32, 4, 84, 84))

    predictions = reward_model.forward(minibatch)
    assert predictions.shape


def test_reward_modelling_process_puts_initial_reward_model_in_queue(
    mocker, reward_modelling_process
):
    # Given
    reward_model = mocker.Mock()
    reward_model_queue = mocker.Mock()
    reward_model_queue.put = mocker.Mock()

    # When
    reward_modeller = reward_modelling_process(
        reward_model=reward_model, reward_model_queue=reward_model_queue
    )

    # Then
    assert reward_modeller.reward_model == reward_model
    assert reward_modeller.reward_model_queue == reward_model_queue
    reward_modeller.reward_model_queue.put.assert_called_once_with(reward_model)


def test_reward_modelling_process_run_gets_all_preferences_from_queue(
    mocker, runnable_reward_modelling_process
):
    # Given
    preference_queue = mocker.Mock()
    preference_queue.empty = mocker.Mock(
        side_effect=[False, False, False, False, False, True]
    )
    preference1 = mocker.Mock()
    preference2 = mocker.Mock()
    preference3 = mocker.Mock()
    preference4 = mocker.Mock()
    preference5 = mocker.Mock()
    preference_queue.get = mocker.Mock(
        side_effect=[preference1, preference2, preference3, preference4, preference5]
    )

    training_buffer = mocker.Mock()
    training_buffer.add = mocker.Mock()
    training_buffer.__len__ = mocker.Mock(return_value=1)
    evaluation_buffer = mocker.Mock()
    evaluation_buffer.add = mocker.Mock()
    evaluation_buffer.__len__ = mocker.Mock(return_value=1)

    # When
    reward_modelling_process = runnable_reward_modelling_process(
        preference_queue=preference_queue,
        training_buffer=training_buffer,
        evaluation_buffer=evaluation_buffer,
        use_for_training=[1, 1, 1, 0, 1],
    )
    reward_modelling_process.run()

    # Then
    assert preference_queue.get.call_count == 5
    assert training_buffer.add.call_count == 4
    preferences_added_to_training_buffer = [
        call_args.args[0] for call_args in training_buffer.add.call_args_list
    ]
    assert preferences_added_to_training_buffer[0] == preference1
    assert preferences_added_to_training_buffer[1] == preference2
    assert preferences_added_to_training_buffer[2] == preference3
    assert preferences_added_to_training_buffer[3] == preference5
    assert evaluation_buffer.add.call_count == 1
    assert evaluation_buffer.add.call_args.args[0] == preference4


def test_reward_modelling_process_run_does_not_train_when_not_enough_comparisons_are_available(
    mocker, runnable_reward_modelling_process
):
    training_buffer = mocker.Mock()
    training_buffer.__len__ = mocker.Mock(return_value=399)
    evaluation_buffer = mocker.Mock()
    evaluation_buffer.__len__ = mocker.Mock(return_value=100)

    reward_modelling_process = runnable_reward_modelling_process(
        training_buffer=training_buffer, evaluation_buffer=evaluation_buffer
    )
    reward_modelling_process.train_reward_model_for_one_epoch = mocker.Mock()

    reward_modelling_process.run()

    assert reward_modelling_process.train_reward_model_for_one_epoch.call_count == 0


@pytest.mark.parametrize(
    "has_completed_pretraining, trained_for_epochs", [(False, 200), (True, 1)]
)
def test_reward_modelling_process_run_trains_reward_model_when_enough_preferences_are_available(
    has_completed_pretraining,
    trained_for_epochs,
    mocker,
    runnable_reward_modelling_process,
):
    training_buffer = mocker.Mock()
    training_buffer.__len__ = mocker.Mock(return_value=400)
    evaluation_buffer = mocker.Mock()
    evaluation_buffer.__len__ = mocker.Mock(return_value=100)
    reward_model = mocker.Mock()
    reward_model.has_completed_pretraining = has_completed_pretraining
    reward_model_queue = mocker.Mock()
    reward_model_queue.put = mocker.Mock()

    reward_modelling_process = runnable_reward_modelling_process(
        reward_model=reward_model,
        training_buffer=training_buffer,
        evaluation_buffer=evaluation_buffer,
        reward_model_queue=reward_model_queue,
    )
    reward_modelling_process.train_reward_model_for_one_epoch = mocker.Mock()

    reward_modelling_process.run()

    assert (
        reward_modelling_process.train_reward_model_for_one_epoch.call_count
        == trained_for_epochs
    )
    reward_model_queue.put.assert_called_with(reward_model)


def test_reward_modelling_process_can_train_reward(mocker, reward_modelling_process):
    # Given
    observations1 = mocker.Mock()
    segment1 = mocker.Mock()
    segment1.get_observations = mocker.Mock(return_value=observations1)
    observations2 = mocker.Mock()
    segment2 = mocker.Mock()
    segment2.get_observations = mocker.Mock(return_value=observations2)

    preference = Preference(segment1=segment1, segment2=segment2, mu=1.0)

    training_buffer = mocker.Mock()
    training_buffer.get_minibatches = mocker.Mock(return_value=[[preference]])

    obs_tensor1 = mocker.Mock()
    ons_tensor2 = mocker.Mock()

    create_tensor = torch.tensor

    def mock_tensor(items):
        if items == observations1:
            return obs_tensor1
        if items == observations2:
            return ons_tensor2
        return create_tensor(items)

    mocker.patch("src.rewardmodelling.torch.tensor", side_effect=mock_tensor)

    model_rewards1 = create_tensor([0.1, 0.2, 0.3])
    model_rewards2 = create_tensor([0.4, 0.5, 0.6])

    def mock_reward_estimate(items):
        if items == obs_tensor1:
            return model_rewards1
        if items == ons_tensor2:
            return model_rewards2
        return create_tensor(items)

    reward_model = mocker.Mock(side_effect=mock_reward_estimate)

    loss = mocker.Mock()
    loss.backward = mocker.Mock()
    mocker.patch("src.rewardmodelling.nn.CrossEntropyLoss.__call__", return_value=loss)

    reward_model_optimizer = mocker.Mock()
    reward_model_optimizer.zero_grad = mocker.Mock()
    reward_model_optimizer.step = mocker.Mock()

    reward_modeller = reward_modelling_process(
        training_buffer=training_buffer,
        reward_model=reward_model,
        reward_model_optimizer=reward_model_optimizer,
    )
    reward_modeller.evaluate_model = mocker.Mock()

    # When
    reward_modeller.train_reward_model_for_one_epoch()

    # Then
    ce_kwargs = src.rewardmodelling.nn.CrossEntropyLoss.__call__.call_args.kwargs
    assert torch.equal(ce_kwargs["input"], create_tensor([[0.6, 1.5]]))
    assert torch.equal(ce_kwargs["target"], create_tensor([[0.0, 1.0]]))
    loss.backward.assert_called_once()
    reward_modeller.reward_model_optimizer.zero_grad.assert_called_once()
    reward_modeller.reward_model_optimizer.step.assert_called_once()
    reward_modeller.evaluate_model.assert_called_once()
