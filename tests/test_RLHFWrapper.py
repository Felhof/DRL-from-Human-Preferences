import multiprocessing
from unittest.mock import call

import pytest

from src.RLHF import RLHFWrapper, start_collecting_trajectories, start_reward_modelling


@pytest.fixture
def rlhf(mocker):
    environment = mocker.Mock()

    rlhf_wrapper = RLHFWrapper(environment=environment)
    return rlhf_wrapper


@pytest.fixture
def rlhf_with_reset(rlhf):
    def _create_rlhf_with_reset(
        observation=None, reward=0.0, terminated=False, truncated=False, info=None
    ):
        rlhf_wrapper = rlhf()
        rlhf_wrapper.environment.reset.return_value = (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )
        return rlhf_wrapper

    return _create_rlhf_with_reset


@pytest.fixture
def rlhf_with_step(rlhf):
    def _create_rlhf_with_step(
        observation=None, reward=0.0, terminated=False, truncated=False, info=None
    ):
        rlhf_wrapper = rlhf()
        rlhf_wrapper.environment.step.return_value = (
            observation,
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
        action=None, new_model_available=False, **kwargs
    ) -> RLHFWrapper:
        observation = kwargs.get("observation", None)

        def mock_reward_model(input_action, input_observation):
            if input_action == action and input_observation == observation:
                return 1.0
            return -1.0

        new_reward_model.get_reward = mocker.Mock(
            side_effect=mock_reward_model if new_model_available else -1.0
        )
        old_reward_model.get_reward = mocker.Mock(
            return_value=-1.0 if new_model_available else mock_reward_model
        )
        reward_model_queue.empty = mocker.Mock(return_value=not new_model_available)
        reward_model_queue.get = mocker.Mock(
            return_value=new_reward_model if new_model_available else old_reward_model
        )

        rlhf_wrapper = rlhf_with_step(**kwargs)

        return rlhf_wrapper

    return _create_rlhf_with_reward_model_queue


def test_start_rlhf_starts_other_processes(mocker):
    # Given
    reward_modelling_process = mocker.Mock()
    trajectory_collecting_process = mocker.Mock()

    def mock_process_from_target(target, args):
        if target == start_reward_modelling:
            return reward_modelling_process
        if target == start_collecting_trajectories:
            return trajectory_collecting_process
        return None

    mocker.patch("multiprocessing.Process", side_effect=mock_process_from_target)

    # When
    rlhf = RLHFWrapper(environment=mocker.Mock())
    rlhf.start_rlhf()

    # Then
    assert multiprocessing.Process.call_count == 2
    call_args_list = multiprocessing.Process.call_args_list
    assert (
        call(target=start_reward_modelling, args=rlhf.reward_model_queue)
        in call_args_list
    )
    assert (
        call(target=start_collecting_trajectories, args=rlhf.trajectory_queue)
        in call_args_list
    )
    reward_modelling_process.start.assert_called_once()
    trajectory_collecting_process.start.assert_called_once()


def test_step_when_no_new_model_available_returns_correct_values(
    mocker,
    rlhf_with_reward_model_queue,
):
    # Given
    action = mocker.Mock()
    info = {"k": "v"}
    observation = mocker.Mock()

    rlhf_wrapper = rlhf_with_reward_model_queue(
        action=action, info=info, new_model_available=False, observation=observation
    )

    # When
    received_obs, reward, terminated, truncated, received_info = rlhf_wrapper.step(
        rlhf_with_reward_model_queue.action
    )

    # Then
    assert received_obs == observation
    assert reward == 1.0
    assert not terminated
    assert not truncated
    assert received_info == info
    assert rlhf_wrapper.reward_model == rlhf_with_reward_model_queue.old_reward_model


def test_step_when_new_model_available_returns_correct_values_and_updates_reward_model(
    mocker,
    rlhf_with_reward_model_queue,
):
    # Given
    action = mocker.Mock()
    info = {"k": "v"}
    observation = mocker.Mock()

    rlhf_wrapper = rlhf_with_reward_model_queue(
        action=action, info=info, new_model_available=True, observation=observation
    )

    # When
    received_obs, reward, terminated, truncated, received_info = rlhf_wrapper.step(
        rlhf_with_reward_model_queue.action
    )

    # Then
    assert received_obs == observation
    assert reward == 1.0
    assert not terminated
    assert not truncated
    assert received_info == info
    assert rlhf_wrapper.reward_model == rlhf_with_reward_model_queue.new_reward_model


def test_step_updates_current_observation(mocker, rlhf_with_step):
    # Given
    action = mocker.Mock()
    info = {"k": "v"}
    observation = mocker.Mock()

    rlhf_wrapper = rlhf_with_step(info=info, observation=observation)

    # When
    rlhf_wrapper.step(action)

    # Then
    assert rlhf_wrapper.current_observation == observation


def test_reset_returns_correct_values(mocker):
    # Given
    expected_info = {"k": "v"}
    expected_observation = mocker.Mock()
    expected_reward = 1.0
    expected_truncated = False
    expected_terminated = False

    rlhf_wrapper = rlhf_with_reset(
        info=expected_info,
        reward=expected_reward,
        truncated=expected_truncated,
        observation=expected_observation,
        terminated=expected_terminated,
    )

    # When
    (
        received_obs,
        received_reward,
        received_truncated,
        received_terminated,
        received_info,
    ) = rlhf_wrapper.reset()

    # Then
    assert received_obs == expected_observation
    assert received_reward == expected_reward
    assert received_truncated == expected_truncated
    assert received_terminated == received_truncated
    assert received_info == received_info


def test_reset_stores_current_observation(mocker, rlhf_with_step):
    # Given
    info = {"k": "v"}
    observation = mocker.Mock()

    rlhf_wrapper = rlhf_with_reset(info=info, observation=observation)

    # When
    rlhf_wrapper.reset()

    # Then
    assert rlhf_wrapper.current_observation == observation
