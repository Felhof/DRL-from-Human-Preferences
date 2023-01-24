from src.preferences import Segment


def test_segment_stores_data_correctly(mocker):
    # Given
    observation1 = mocker.Mock()
    action1 = mocker.Mock()
    observation2 = mocker.Mock()
    action2 = mocker.Mock()

    segment_data = [
        (observation1, action1),
        (observation2, action2),
    ]

    # When
    segment = Segment(segment_data)

    # Then
    assert segment.data == segment_data


def test_segment_get_observations_gets_only_segment_observations(mocker):
    # Given
    observation1 = mocker.Mock()
    observation2 = mocker.Mock()
    observation3 = mocker.Mock()

    segment_data = [
        (observation1, mocker.Mock()),
        (observation2, mocker.Mock()),
        (observation3, mocker.Mock()),
    ]

    # When
    segment = Segment(segment_data)

    # Then
    assert segment.get_observations() == [observation1, observation2, observation3]
