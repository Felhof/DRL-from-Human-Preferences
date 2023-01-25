import pytest
from typing import List
from unittest.mock import call

from src.preferences import FeedbackCollectionProcess, Segment, SegmentDB, SEGMENT_LENGTH
import src.preferences


@pytest.fixture
def feedback_collection_process(mocker):
    def _create_feedback_collection_process(segment_queue=None, segment_db=None):
        mocker.patch("src.preferences.SegmentDB", return_value=segment_db)
        mocker.patch("src.preferences.PreferenceElicitationThread.run")
        mocker.patch("src.preferences.Queue", return_value=segment_queue)
        feedback_collection = FeedbackCollectionProcess()

        return feedback_collection

    return _create_feedback_collection_process


@pytest.fixture
def trajectory_queue_with_items(mocker):
    def _create_trajectory_queue(items: List):
        items = ["END"] + items

        def empty():
            return len(items) == 0

        def get():
            assert len(items) > 0, "Can only call get on Queue if the there are items in the Queue!"
            item = items.pop()
            return item

        trajectory_queue = mocker.Mock()
        trajectory_queue.empty = mocker.Mock(side_effect=empty)
        trajectory_queue.get = mocker.Mock(side_effect=get)

        return trajectory_queue

    return _create_trajectory_queue


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


def test_feedback_collection_process_run_runs_until_receiving_end_message(
        feedback_collection_process, trajectory_queue_with_items
):
    trajectory_queue = trajectory_queue_with_items(["A", "B", "C"])

    feedback_collection = feedback_collection_process()
    feedback_collection.run(trajectory_queue)

    assert feedback_collection.trajectory_queue.get.call_count == 4


def test_feedback_collection_process_run_initializes_segment_db(
        feedback_collection_process, trajectory_queue_with_items
):
    trajectory_queue = trajectory_queue_with_items([])

    feedback_collection = feedback_collection_process()
    feedback_collection.run(trajectory_queue)

    assert src.preferences.SegmentDB.call_count == 1


def test_feedback_collection_process_run_starts_preference_elicitation_thread(
        feedback_collection_process, mocker, trajectory_queue_with_items
):
    segment_queue = mocker.Mock()
    trajectory_queue = trajectory_queue_with_items([])

    feedback_collection = feedback_collection_process(segment_queue=segment_queue)
    feedback_collection.run(trajectory_queue)

    assert src.preferences.PreferenceElicitationThread.run.call_count == 1
    src.preferences.PreferenceElicitationThread.run.assert_called_with(
        queue=segment_queue
    )


def test_feedback_collection_process_stores_segments_from_available_trajectories(
        feedback_collection_process, mocker, trajectory_queue_with_items
):
    segment_db = mocker.Mock()
    segment_db.store_segments = mocker.Mock()
    trajectory = [(mocker.Mock(), mocker.Mock()) for _ in range(SEGMENT_LENGTH * 3)]
    segment1 = trajectory[:SEGMENT_LENGTH]
    segment2 = trajectory[SEGMENT_LENGTH:SEGMENT_LENGTH * 2]
    segment3 = trajectory[SEGMENT_LENGTH * 2:]
    trajectory_queue = trajectory_queue_with_items([trajectory])

    feedback_collection = feedback_collection_process(segment_db=segment_db)
    feedback_collection.run(trajectory_queue)

    assert segment_db.store_segments.call_count == 1
    segment_db.store_segments.assert_called_with(
        [segment1, segment2, segment3]
    )
