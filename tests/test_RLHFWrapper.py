import numpy as np
import pytest
import torch

import src.rewardmodelling
from src.RLHF import RLHFWrapper
import src.RLHF


@pytest.fixture
def rlhf(mocker):
    environment = mocker.Mock()

    rlhf_wrapper = RLHFWrapper(environment=environment)
    rlhf_wrapper.logger = mocker.Mock()
    return rlhf_wrapper


@pytest.fixture
def rlhf_with_reset(rlhf):
    def _create_rlhf_with_reset(current_observation=None, next_observation=None):
        rlhf_wrapper = rlhf
        rlhf_wrapper.current_observation = current_observation
        rlhf_wrapper.environment.reset.return_value = next_observation
        return rlhf_wrapper

    return _create_rlhf_with_reset


@pytest.fixture
def rlhf_with_step(mocker, rlhf):
    def _create_rlhf_with_step(
        action=None,
        current_observation=None,
        next_observation=None,
        reward=0.0,
        done=False,
        info=None,
    ):
        rlhf_wrapper = rlhf
        rlhf_wrapper.current_observation = current_observation
        rlhf_wrapper.environment.step = mocker.Mock(
            side_effect=lambda a: (
                next_observation,
                reward,
                done,
                info,
            )
            if a == action
            else mocker.Mock()
        )
        return rlhf_wrapper

    return _create_rlhf_with_step


@pytest.fixture
def rlhf_with_reward_model_queue(mocker, rlhf_with_step):
    reward_model_queue = mocker.Mock()

    def _create_rlhf_with_reward_model_queue(
        new_model_available=False, np_observation=mocker.Mock(), reward=1.0, **kwargs
    ) -> RLHFWrapper:

        next_observation = kwargs.get("next_observation", None)

        tensor_observation = mocker.Mock()

        create_np_array = np.array

        def mock_array_creation(o):
            return np_observation if o == next_observation else create_np_array(o)

        mocker.patch(
            "src.RLHF.np.array",
            side_effect=mock_array_creation,
        )

        create_tensor = torch.tensor

        def mock_tensor_creation(o, **kwargs):
            return tensor_observation if o == np_observation else create_tensor(o)

        mocker.patch("src.RLHF.torch.tensor", side_effect=mock_tensor_creation)

        def mock_reward_model(input_observation):
            if input_observation == tensor_observation:
                return torch.tensor(reward)
            return torch.tensor(-reward)

        def mock_new_reward_model(input_observation):
            if new_model_available:
                return mock_reward_model(input_observation)
            return torch.tensor(-reward)

        def mock_old_reward_model(input_observation):
            if not new_model_available:
                return mock_reward_model(input_observation)
            return torch.tensor(-reward)

        new_reward_model = mocker.Mock(side_effect=mock_new_reward_model)
        old_reward_model = mocker.Mock(side_effect=mock_old_reward_model)
        reward_model_queue.qsize = mocker.Mock(
            return_value=1 if new_model_available else 0
        )
        reward_model_queue.get = mocker.Mock(return_value=new_reward_model)

        rlhf_wrapper = rlhf_with_step(**kwargs)
        rlhf_wrapper.reward_model_queue = reward_model_queue
        rlhf_wrapper.reward_model = old_reward_model

        return rlhf_wrapper

    return _create_rlhf_with_reward_model_queue


def test_full_rlhf_executes_all_steps(mocker, rlhf_with_reward_model_queue):
    # Given
    rlhf = rlhf_with_reward_model_queue()
    rlhf._start_other_processes = mocker.Mock()
    rlhf._collect_initial_preferences = mocker.Mock()
    rlhf._wait_for_pretraining_of_reward_model = mocker.Mock()

    # When
    rlhf.full_rlhf(preference_target="some_file")

    # Then
    rlhf._start_other_processes.assert_called_once_with(
        mode="full_rlhf", preference_target="some_file"
    )
    rlhf._collect_initial_preferences.assert_called_once()
    rlhf._wait_for_pretraining_of_reward_model.assert_called_once()


def test_only_collect_initial_preferences_collects_initial_preferences_and_then_stops(
    mocker, rlhf_with_reward_model_queue
):
    # Given
    rlhf = rlhf_with_reward_model_queue()
    rlhf._start_other_processes = mocker.Mock()
    rlhf._collect_initial_preferences = mocker.Mock()
    rlhf._wait_for_pretraining_of_reward_model = mocker.Mock()
    rlhf.stop = mocker.Mock()

    # When
    rlhf.only_collect_initial_preferences(preference_target="some_file")

    # Then
    rlhf._start_other_processes.assert_called_once_with(
        mode="collect_initial_preferences", preference_target="some_file"
    )
    rlhf._collect_initial_preferences.assert_called_once()
    assert rlhf._wait_for_pretraining_of_reward_model.call_count == 0
    rlhf.stop.assert_called_once()


def test_can_start_with_pretraining(mocker, rlhf_with_reward_model_queue):
    # Given
    rlhf = rlhf_with_reward_model_queue()
    rlhf._start_other_processes = mocker.Mock()
    rlhf._collect_initial_preferences = mocker.Mock()
    rlhf._wait_for_pretraining_of_reward_model = mocker.Mock()
    rlhf.stop = mocker.Mock()

    # When
    rlhf.rlhf_starting_with_pretraining(
        preference_source="source_file", preference_target="target_file"
    )

    # Then
    rlhf._start_other_processes.assert_called_once_with(
        mode="start_with_pretraining",
        preference_target="target_file",
        preference_source="source_file",
    )
    rlhf._wait_for_pretraining_of_reward_model.assert_called_once()
    assert rlhf._collect_initial_preferences.call_count == 0


def test_collect_initial_preferences_takes_random_actions_until_enough_initial_preferences_are_collected(
    mocker, rlhf_with_reward_model_queue
):
    # Given
    rlhf = rlhf_with_reward_model_queue()
    rlhf.reset = mocker.Mock()
    rlhf.collected_initial_preferences_queue.qsize = mocker.Mock(
        side_effect=[0, 0, 0, 1]
    )
    rlhf.environment.action_space.sample = mocker.Mock(side_effect=[1, 2, 3])
    rlhf._step = mocker.Mock(return_value=(None, None, None, None))

    # When
    rlhf._collect_initial_preferences()

    # Then
    rlhf.reset.assert_called_once()
    rlhf._step.assert_has_calls([mocker.call(1), mocker.call(2), mocker.call(3)])


def test_rlhf_wrapper_can_start_other_processes(mocker, rlhf_with_reward_model_queue):
    # Given
    log_listener = mocker.Mock()
    reward_modelling_process = mocker.Mock()
    feedback_collecting_process = mocker.Mock()

    mocker.patch("src.RLHF.LogListener", return_value=log_listener)
    mocker.patch(
        "src.RLHF.RewardModellingProcess", return_value=reward_modelling_process
    )
    mocker.patch(
        "src.RLHF.FeedbackCollectionProcess", return_value=feedback_collecting_process
    )

    log_queue = mocker.Mock()
    preference_queue = mocker.Mock()
    mocker.patch("src.RLHF.Queue", side_effect=[log_queue, preference_queue])

    mocker.patch("src.RLHF.logging.getLogger", return_value=mocker.Mock())

    # When
    rlhf = rlhf_with_reward_model_queue()
    rlhf._start_other_processes()

    # Then
    log_listener.start.assert_called_once()
    src.RLHF.RewardModellingProcess.assert_called_once_with(
        preference_queue=preference_queue,
        reward_model_queue=rlhf.reward_model_queue,
        stop_queue=rlhf.stop_reward_modelling_queue,
        preference_source="",
        preference_target="",
        collected_initial_preferences_queue=rlhf.collected_initial_preferences_queue,
        mode="full_rlhf",
    )
    src.RLHF.FeedbackCollectionProcess.assert_called_once_with(
        preference_queue=preference_queue,
        trajectory_queue=rlhf.trajectory_queue,
        stop_queue=rlhf.stop_feedback_collecting_queue,
    )
    reward_modelling_process.start.assert_called_once()
    feedback_collecting_process.start.assert_called_once()


def test_step_when_no_new_model_available_returns_correct_values(
    mocker,
    rlhf_with_reward_model_queue,
):
    # Given
    action = mocker.Mock()
    current_observation = mocker.Mock()
    expected_info = {"k": "v"}
    expected_observation = mocker.Mock()
    expected_reward = 1.0

    rlhf_wrapper = rlhf_with_reward_model_queue(
        action=action,
        info=expected_info,
        new_model_available=False,
        current_observation=current_observation,
        next_observation=expected_observation,
        reward=expected_reward,
    )

    # When
    (
        received_obs,
        received_reward,
        done,
        received_info,
    ) = rlhf_wrapper.step(action)

    # Then
    assert received_obs == expected_observation
    assert received_reward == expected_reward
    assert not done
    assert received_info == expected_info


def test_step_when_new_model_available_returns_correct_values_and_updates_reward_model(
    mocker,
    rlhf_with_reward_model_queue,
):
    # Given
    action = mocker.Mock()
    current_observation = mocker.Mock()
    expected_info = {"k": "v"}
    expected_observation = mocker.Mock()
    expected_reward = 1.0

    rlhf_wrapper = rlhf_with_reward_model_queue(
        action=action,
        info=expected_info,
        new_model_available=True,
        current_observation=current_observation,
        next_observation=expected_observation,
        reward=expected_reward,
    )

    # When
    received_obs, reward, done, received_info = rlhf_wrapper.step(action)

    # Then
    assert received_obs == expected_observation
    assert reward == expected_reward
    assert not done
    assert received_info == expected_info


def test_step_updates_current_observation(mocker, rlhf_with_reward_model_queue):
    # Given
    action = mocker.Mock()
    info = {"k": "v"}
    next_observation = mocker.Mock()
    expected_observation = mocker.Mock()

    rlhf_wrapper = rlhf_with_reward_model_queue(
        action=action,
        info=info,
        next_observation=next_observation,
        np_observation=expected_observation,
    )

    # When
    rlhf_wrapper.step(action)

    # Then
    assert rlhf_wrapper.current_observation == expected_observation


def test_step_adds_current_observation_and_action_to_current_trajectory(
    mocker, rlhf_with_reward_model_queue
):
    # Given
    action = mocker.Mock()
    info = {"k": "v"}
    current_observation = mocker.Mock()
    next_observation = mocker.Mock()

    rlhf_wrapper = rlhf_with_reward_model_queue(
        action=action,
        info=info,
        current_observation=current_observation,
        next_observation=next_observation,
    )

    # When
    rlhf_wrapper.step(action)

    # Then
    assert rlhf_wrapper.current_trajectory[-1] == (current_observation, action)


@pytest.mark.parametrize(
    "trajectory_queue_is_full, expected_put_call_count", [(True, 0), (False, 1)]
)
def test_step_when_done_sends_current_trajectory_to_feedback_process(
    trajectory_queue_is_full,
    expected_put_call_count,
    mocker,
    rlhf_with_reward_model_queue,
):
    # Given
    action = mocker.Mock()
    info = {"k": "v"}
    current_observation = mocker.Mock()
    next_observation = mocker.Mock()
    trajectory_queue = mocker.Mock()
    trajectory_queue.put = mocker.Mock()
    trajectory_queue.full = mocker.Mock(return_value=trajectory_queue_is_full)
    current_trajectory = mocker.Mock()
    current_trajectory.__len__ = mocker.Mock(return_value=50)

    rlhf_wrapper = rlhf_with_reward_model_queue(
        action=action,
        info=info,
        current_observation=current_observation,
        next_observation=next_observation,
        done=True,
    )

    rlhf_wrapper.current_trajectory = current_trajectory
    rlhf_wrapper.trajectory_queue = trajectory_queue

    # When
    rlhf_wrapper.step(action)

    # Then
    assert rlhf_wrapper.trajectory_queue.put.call_count == expected_put_call_count
    if not trajectory_queue_is_full:
        assert rlhf_wrapper.trajectory_queue.put.called_with(current_trajectory)
    assert rlhf_wrapper.current_trajectory == []


def test_reset_returns_correct_values(mocker, rlhf_with_reset):
    # Given
    expected_observation = mocker.Mock()

    rlhf_wrapper = rlhf_with_reset(
        next_observation=expected_observation,
    )

    # When
    next_observation = rlhf_wrapper.reset()

    # Then
    assert next_observation == expected_observation


def test_reset_updates_current_observation(mocker, rlhf_with_reset):
    # Given
    next_observation = mocker.Mock()

    rlhf_wrapper = rlhf_with_reset(next_observation=next_observation)

    # When
    rlhf_wrapper.reset()

    # Then
    assert rlhf_wrapper.current_observation == next_observation


def test_reset_clears_trajectory_queue(mocker, rlhf_with_reset):
    # Given
    next_observation = mocker.Mock()
    current_trajectory = mocker.Mock()

    rlhf_wrapper = rlhf_with_reset(next_observation=next_observation)
    rlhf_wrapper.current_trajectory = current_trajectory

    # When
    rlhf_wrapper.reset()

    # Then
    assert rlhf_wrapper.current_trajectory == []
