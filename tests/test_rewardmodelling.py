import numpy as np

from src.rewardmodelling import PreferenceBuffer


def test_preference_buffer_can_add_new_items_up_to_buffer_size_and_loops_afterwards(mocker):
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
    mocker.patch("src.rewardmodelling.np.random.choice", return_value=np.array([1, 4, 2]))

    # When
    preference_buffer = PreferenceBuffer()

    for p in [p1, p2, p3, p4, p5]:
        preference_buffer.add(p)

    minibatch = preference_buffer.get_minibatch(n=3)

    # Then
    assert minibatch == [p2, p5, p3]
