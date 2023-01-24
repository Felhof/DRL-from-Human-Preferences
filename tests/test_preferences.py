from unittest.mock import call

from src.preferences import FeedbackCollectionProcess, Segment, SegmentDB
import src.preferences


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

    segment = Segment(segment_data)

    # Then
    assert segment.get_observations() == [observation1, observation2, observation3]


def test_segmentdb_store_segments_adds_segments_pairs_to_back_of_deque(mocker):
    segment1 = mocker.Mock()
    segment2 = mocker.Mock()
    segment3 = mocker.Mock()

    segment_db = SegmentDB()

    assert len(segment_db.segments) == 0

    segment_db.store_segments([segment1, segment2, segment3])

    assert len(segment_db.segments) == 3
    assert segment_db.segments[0] == segment1
    assert segment_db.segments[1] == segment2
    assert segment_db.segments[2] == segment3


def test_segmentdb_query_segment_pairs_returns_n_pairs_of_unique_segments(mocker):
    segment1 = mocker.Mock()
    segment2 = mocker.Mock()
    segment3 = mocker.Mock()
    segment4 = mocker.Mock()

    segment_db = SegmentDB()

    segment_db.store_segments([segment1, segment2, segment3, segment4])

    segment_pair1, segment_pair2 = segment_db.query_segment_pairs(2)

    assert segment_pair1[0] not in segment_pair2
    assert segment_pair1[1] not in segment_pair2
    assert segment1 in segment_pair1 or segment_pair2
    assert segment2 in segment_pair1 or segment_pair2
    assert segment3 in segment_pair1 or segment_pair2
    assert segment4 in segment_pair1 or segment_pair2


def test_feedback_collection_process_run_initializes_segment_db(mocker):
    mocker.patch("src.preferences.SegmentDB")

    feedback_collection = FeedbackCollectionProcess()
    feedback_collection.run()

    assert src.preferences.SegmentDB.call_count == 1


def test_feedback_collection_process_run_starts_preference_elicitation_thread(mocker):
    segment_queue = mocker.Mock()
    mocker.patch("src.preferences.PreferenceElicitationThread.run")

    feedback_collection = FeedbackCollectionProcess()
    feedback_collection.segment_queue = segment_queue
    feedback_collection.run()

    assert src.preferences.PreferenceElicitationThread.run.call_count == 1
    src.preferences.PreferenceElicitationThread.run.assert_called_with(queue=segment_queue)
