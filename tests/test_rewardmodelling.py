import os

import numpy as np
import pytest
import torch

import src.rewardmodelling
from src.preferences import Preference, Segment
from src.rewardmodelling import PreferenceBuffer, RewardModel, RewardModellingProcess


@pytest.fixture()
def cleanup_preference_data() -> None:
    yield
    os.remove(f"../data/preferences.ptk")


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
        collected_initial_preferences_queue=None,
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

        if collected_initial_preferences_queue is None:
            collected_initial_preferences_queue = mocker.Mock()

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

        mocker.patch("src.preferences.logging.getLogger", return_value=mocker.Mock())

        reward_modeller = RewardModellingProcess(
            preference_queue=preference_queue,
            reward_model_queue=reward_model_queue,
            stop_queue=stop_queue,
            collected_initial_preferences_queue=collected_initial_preferences_queue,
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
            preference_queue.qsize = mocker.Mock(return_value=0)
            kwargs["preference_queue"] = preference_queue

        if "stop_queue" not in kwargs:
            stop_queue = mocker.Mock()
            stop_queue.qsize = mocker.Mock(side_effect=[0, 1])
            stop_queue.get = mocker.Mock(side_effect=[True])
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


def test_preference_buffer_can_save_and_load(cleanup_preference_data, mocker):
    mocker.patch("src.preferences.SEGMENT_LENGTH", 5)

    def create_random_trajectory(n=5):
        return [(np.random.rand(2), np.random.rand(1)) for _ in range(n)]

    segments = [Segment(create_random_trajectory()) for _ in range(6)]

    preference1 = Preference(segment1=segments[0], segment2=segments[1], mu=0.0)
    preference2 = Preference(
        segment1=segments[2],
        segment2=segments[3],
        mu=0.5,
    )
    preference3 = Preference(segment1=segments[4], segment2=segments[5], mu=1.0)

    preference_buffer1 = PreferenceBuffer(buffer_size=500)
    preference_buffer1.add(preference1)
    preference_buffer1.add(preference2)
    preference_buffer1.add(preference3)
    preference_buffer1.save_to_file()

    preference_buffer2 = PreferenceBuffer()
    preference_buffer2.load_from_file()

    assert preference_buffer1.preferences == preference_buffer2.preferences
    assert (
        preference_buffer1.number_of_preferences
        == preference_buffer2.number_of_preferences
    )
    assert preference_buffer1.idx == preference_buffer2.idx
    assert preference_buffer1.buffer_size == preference_buffer2.buffer_size


def test_reward_model_maps_observation_to_scalar():
    reward_model = RewardModel()

    single_frame = torch.rand((4, 84, 84))
    prediction = reward_model.forward(single_frame)

    assert prediction.shape == torch.Size([1])

    minibatch = torch.rand((32, 4, 84, 84))

    predictions = reward_model.forward(minibatch)
    assert predictions.shape


def test_reward_modelling_process_can_collect_initial_preferences(
    mocker, runnable_reward_modelling_process
):
    # Given
    preference_queue = mocker.Mock()
    preference_queue.qsize = mocker.Mock(side_effect=[5, 4, 3, 2, 1, 0])

    preferences = [mocker.Mock() for _ in range(5)]
    preference_queue.get = mocker.Mock(side_effect=preferences)

    use_for_training = [1, 0, 1, 1, 1]

    cip_queue = mocker.Mock()

    reward_modelling_process = runnable_reward_modelling_process(
        preference_queue=preference_queue,
        collected_initial_preferences_queue=cip_queue,
        use_for_training=use_for_training,
    )
    reward_modelling_process._save_preference_buffers = mocker.Mock()
    reward_modelling_process._number_of_stored_preferences = mocker.Mock(
        side_effect=[0, 1, 2, 3, 4, 5]
    )

    # When
    reward_modelling_process._collect_initial_preferences(n=5)

    # Then
    training_buffer_add_calls = (
        reward_modelling_process.training_buffer.add.call_args_list
    )
    evaluation_buffer_add_calls = (
        reward_modelling_process.evaluation_buffer.add.call_args_list
    )
    assert training_buffer_add_calls[0].args[0] == preferences[0]
    assert training_buffer_add_calls[1].args[0] == preferences[2]
    assert training_buffer_add_calls[2].args[0] == preferences[3]
    assert training_buffer_add_calls[3].args[0] == preferences[4]
    assert evaluation_buffer_add_calls[0].args[0] == preferences[1]
    reward_modelling_process._save_preference_buffers.assert_called_once()


def test_reward_modelling_process_can_pretrain_reward_model(
    mocker, runnable_reward_modelling_process
):
    # Given
    reward_model_queue = mocker.Mock()
    reward_model_queue.put = mocker.Mock()

    reward_modelling_process = runnable_reward_modelling_process(
        reward_model_queue=reward_model_queue
    )
    reward_modelling_process.preference_source = "test"
    reward_modelling_process.train_reward_model_for_one_epoch = mocker.Mock()
    reward_modelling_process._collect_initial_preferences = mocker.Mock()
    reward_modelling_process._load_preference_buffers = mocker.Mock()
    reward_modelling_process._reward_model_training_loop = mocker.Mock()
    reward_modelling_process.evaluate_model = mocker.Mock()

    # When
    reward_modelling_process.run()

    # Then
    assert reward_modelling_process.train_reward_model_for_one_epoch.call_count == 5
    assert reward_modelling_process.evaluate_model.call_count == 5
    reward_model_queue.put.assert_called_once()


def test_reward_modelling_training_loop_gets_preference_from_queue_and_updates_reward_model(
    mocker, runnable_reward_modelling_process
):
    # Given
    stop_queue = mocker.Mock()
    stop_queue.qsize = mocker.Mock(side_effect=[0] * 32 + [1])
    stop_queue.get = mocker.Mock(return_value=True)

    reward_modelling_process = runnable_reward_modelling_process(stop_queue=stop_queue)

    reward_modelling_process._try_to_store_preference_from_queue_in_buffer = (
        mocker.Mock(return_value=True)
    )
    reward_modelling_process.train_reward_model_for_one_epoch = mocker.Mock()

    # When
    reward_modelling_process._reward_model_training_loop()

    # Then
    assert (
        reward_modelling_process._try_to_store_preference_from_queue_in_buffer.call_count
        == 32
    )
    reward_modelling_process.train_reward_model_for_one_epoch.assert_called_once()


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

    def mock_tensor(items, **kwargs):
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

    mocker.patch("src.rewardmodelling.np.mean")

    # When
    reward_modeller.train_reward_model_for_one_epoch()

    # Then
    ce_kwargs = src.rewardmodelling.nn.CrossEntropyLoss.__call__.call_args.kwargs
    assert torch.equal(ce_kwargs["input"], create_tensor([[0.6, 1.5]]))
    assert torch.equal(ce_kwargs["target"], create_tensor([[0.0, 1.0]]))
    loss.backward.assert_called_once()
    reward_modeller.reward_model_optimizer.zero_grad.assert_called_once()
    reward_modeller.reward_model_optimizer.step.assert_called_once()
