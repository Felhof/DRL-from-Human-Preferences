import multiprocessing
from unittest.mock import call

import pytest

from src.RLHF import RLHFWrapper, start_reward_modelling
import src.RLHF


@pytest.fixture
def rlhf(mocker):
    environment = mocker.Mock()

    rlhf_wrapper = RLHFWrapper(environment=environment)
    return rlhf_wrapper


@pytest.fixture
def rlhf_with_reset(rlhf):
    def _create_rlhf_with_reset(
            current_observation=None, next_observation=None, info=None
    ):
        rlhf_wrapper = rlhf
        rlhf_wrapper.current_observation = current_observation
        rlhf_wrapper.environment.reset.return_value = (
            next_observation,
            info,
        )
        return rlhf_wrapper

    return _create_rlhf_with_reset


@pytest.fixture
def rlhf_with_step(rlhf):
    def _create_rlhf_with_step(
            current_observation=None,
            next_observation=None,
            reward=0.0,
            terminated=False,
            truncated=False,
            info=None,
    ):
        rlhf_wrapper = rlhf
        rlhf_wrapper.current_observation = current_observation
        rlhf_wrapper.environment.step.return_value = (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        )
        return rlhf_wrapper

    return _create_rlhf_with_step


@pytest.fixture
def rlhf_with_reward_model_queue(mocker, rlhf_with_step):
    new_reward_model = mocker.Mock()
    old_reward_model = mocker.Mock()
    reward_model_queue = mocker.Mock()

    def _create_rlhf_with_reward_model_queue(
            action=None, new_model_available=False, reward=1.0, **kwargs
    ) -> RLHFWrapper:
        current_observation = kwargs.get("current_observation", None)

        def mock_reward_model(input_observation, input_action):
            if input_action == action and input_observation == current_observation:
                return reward
            return -reward

        def mock_new_reward_model(input_observation, input_action):
            if new_model_available:
                return mock_reward_model(input_observation, input_action)
            return -reward

        def mock_old_reward_model(input_observation, input_action):
            if not new_model_available:
                return mock_reward_model(input_observation, input_action)
            return -reward

        new_reward_model.get_reward = mocker.Mock(side_effect=mock_new_reward_model)
        old_reward_model.get_reward = mocker.Mock(side_effect=mock_old_reward_model)
        reward_model_queue.empty = mocker.Mock(return_value=not new_model_available)
        reward_model_queue.get = mocker.Mock(return_value=new_reward_model)

        rlhf_wrapper = rlhf_with_step(**kwargs)
        rlhf_wrapper.reward_model_queue = reward_model_queue
        rlhf_wrapper.reward_model = old_reward_model

        return rlhf_wrapper

    return _create_rlhf_with_reward_model_queue


def test_start_rlhf_starts_other_processes(mocker):
    # Given
    reward_modelling_process = mocker.Mock()
    feedback_collecting_process = mocker.Mock()

    mocker.patch("multiprocessing.Process", return_value=reward_modelling_process)
    mocker.patch(
        "src.RLHF.FeedbackCollectionProcess", return_value=feedback_collecting_process
    )

    # When
    rlhf = RLHFWrapper(environment=mocker.Mock())
    rlhf.start_rlhf()

    # Then
    multiprocessing.Process.assert_called_once_with(
        target=start_reward_modelling, args=(rlhf.reward_model_queue,)
    )
    src.RLHF.FeedbackCollectionProcess.assert_called_once_with(rlhf.trajectory_queue)
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
        terminated,
        truncated,
        received_info,
    ) = rlhf_wrapper.step(action)

    # Then
    assert received_obs == expected_observation
    assert received_reward == expected_reward
    assert not terminated
    assert not truncated
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
    received_obs, reward, terminated, truncated, received_info = rlhf_wrapper.step(
        action
    )

    # Then
    assert received_obs == expected_observation
    assert reward == expected_reward
    assert not terminated
    assert not truncated
    assert received_info == expected_info


def test_step_updates_current_observation(mocker, rlhf_with_reward_model_queue):
    # Given
    action = mocker.Mock()
    info = {"k": "v"}
    expected_observation = mocker.Mock()

    rlhf_wrapper = rlhf_with_reward_model_queue(
        action=action, info=info, next_observation=expected_observation
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


def test_step_when_done_sends_current_trajectory_to_feedback_process(
        mocker, rlhf_with_reward_model_queue
):
    # Given
    action = mocker.Mock()
    info = {"k": "v"}
    current_observation = mocker.Mock()
    next_observation = mocker.Mock()
    trajectory_queue = mocker.Mock()
    trajectory_queue.put = mocker.Mock()
    current_trajectory = mocker.Mock()

    rlhf_wrapper = rlhf_with_reward_model_queue(
        action=action,
        info=info,
        current_observation=current_observation,
        next_observation=next_observation,
        terminated=True,
    )

    rlhf_wrapper.current_trajectory = current_trajectory
    rlhf_wrapper.trajectory_queue = trajectory_queue

    # When
    rlhf_wrapper.step(action)

    # Then
    assert rlhf_wrapper.trajectory_queue.put.call_count == 1
    assert rlhf_wrapper.trajectory_queue.put.called_with(current_trajectory)
    assert rlhf_wrapper.current_trajectory == []


def test_reset_returns_correct_values(mocker, rlhf_with_reset):
    # Given
    expected_info = {"k": "v"}
    expected_observation = mocker.Mock()

    rlhf_wrapper = rlhf_with_reset(
        info=expected_info,
        next_observation=expected_observation,
    )

    # When
    (
        next_observation,
        received_info,
    ) = rlhf_wrapper.reset()

    # Then
    assert next_observation == expected_observation
    assert received_info == received_info


def test_reset_updates_current_observation(mocker, rlhf_with_reset):
    # Given
    info = {"k": "v"}
    next_observation = mocker.Mock()

    rlhf_wrapper = rlhf_with_reset(info=info, next_observation=next_observation)

    # When
    rlhf_wrapper.reset()

    # Then
    assert rlhf_wrapper.current_observation == next_observation


def test_reset_clears_trajectory_queue(mocker, rlhf_with_reset):
    # Given
    info = {"k": "v"}
    next_observation = mocker.Mock()
    current_trajectory = mocker.Mock()

    rlhf_wrapper = rlhf_with_reset(info=info, next_observation=next_observation)
    rlhf_wrapper.current_trajectory = current_trajectory

    # When
    rlhf_wrapper.reset()

    # Then
    assert rlhf_wrapper.current_trajectory == []
