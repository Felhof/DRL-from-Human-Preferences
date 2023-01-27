import numpy as np
import torch

from src.rewardmodelling import PreferenceBuffer, RewardModel, RewardModellingProcess


def test_preference_buffer_can_add_new_items_up_to_buffer_size_and_loops_afterwards(
        mocker,
):
    # Given
    mocker.patch("src.rewardmodelling.BUFFER_SIZE", 3)
    p1 = mocker.Mock()
    p2 = mocker.Mock()
    p3 = mocker.Mock()
    p4 = mocker.Mock()

    # When
    preference_buffer = PreferenceBuffer()

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


def test_preference_buffer_can_get_minibatch_of_distinct_entries(mocker):
    # Given
    p1 = mocker.Mock()
    p2 = mocker.Mock()
    p3 = mocker.Mock()
    p4 = mocker.Mock()
    p5 = mocker.Mock()
    mocker.patch(
        "src.rewardmodelling.np.random.choice", return_value=np.array([1, 4, 2])
    )

    # When
    preference_buffer = PreferenceBuffer()

    for p in [p1, p2, p3, p4, p5]:
        preference_buffer.add(p)

    minibatch = preference_buffer.get_minibatch(n=3)

    # Then
    assert minibatch == [p2, p5, p3]


def test_reward_model_maps_observation_to_scalar():
    reward_model = RewardModel()

    single_frame = torch.rand((4, 84, 84))
    prediction = reward_model.forward(single_frame)

    assert prediction.shape == torch.Size([1])

    minibatch = torch.rand((32, 4, 84, 84))

    predictions = reward_model.forward(minibatch)
    assert predictions.shape


def test_reward_modelling_process_puts_initial_reward_model_in_queue(mocker):
    # Given
    preference_queue = mocker.Mock()
    reward_model = mocker.Mock()
    mocker.patch("src.rewardmodelling.RewardModel", return_value=reward_model)
    reward_model_queue = mocker.Mock()
    reward_model_queue.put = mocker.Mock()
    stop_queue = mocker.Mock()

    # When
    reward_modelling_process = RewardModellingProcess(
        preference_queue, reward_model_queue, stop_queue
    )

    # Then
    assert reward_modelling_process.reward_model == reward_model
    assert reward_modelling_process.reward_model_queue == reward_model_queue
    reward_modelling_process.reward_model_queue.put.assert_called_once_with(
        reward_model
    )


def test_reward_modelling_process_run_gets_all_preferences_from_queue(mocker):
    # Given
    preference_queue = mocker.Mock()
    preference_queue.empty = mocker.Mock(side_effect=[False, False, False, False, False, True])
    preference1 = mocker.Mock()
    preference2 = mocker.Mock()
    preference3 = mocker.Mock()
    preference4 = mocker.Mock()
    preference5 = mocker.Mock()
    preference_queue.get = mocker.Mock(side_effect=[preference1, preference2, preference3, preference4, preference5])

    mocker.patch("src.rewardmodelling.RewardModel", return_value=mocker.Mock())
    mocker.patch("src.rewardmodelling.np.random.binomial", side_effect=[1, 1, 1, 0, 1])

    training_buffer = mocker.Mock()
    training_buffer.add = mocker.Mock()
    evaluation_buffer = mocker.Mock()
    evaluation_buffer.add = mocker.Mock()
    mocker.patch("src.rewardmodelling.PreferenceBuffer", side_effect=[training_buffer, evaluation_buffer])

    reward_model_queue = mocker.Mock()
    stop_queue = mocker.Mock()
    stop_queue.empty = mocker.Mock(return_value=False)
    stop_queue.get = mocker.Mock(side_effect=[False, True])

    # When
    reward_modelling_process = RewardModellingProcess(
        preference_queue, reward_model_queue, stop_queue
    )
    reward_modelling_process.run()

    # Then
    assert preference_queue.get.call_count == 5
    assert training_buffer.add.call_count == 4
    preferences_added_to_training_buffer = [call_args.args[0] for call_args in training_buffer.add.call_args_list]
    assert preferences_added_to_training_buffer[0] == preference1
    assert preferences_added_to_training_buffer[1] == preference2
    assert preferences_added_to_training_buffer[2] == preference3
    assert preferences_added_to_training_buffer[3] == preference5
    assert evaluation_buffer.add.call_count == 1
    assert evaluation_buffer.add.call_args.args[0] == preference4
