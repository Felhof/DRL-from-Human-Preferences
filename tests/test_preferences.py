import pytest
from typing import List

from src.preferences import (
    FeedbackCollectionProcess,
    Segment,
    SegmentDB,
    SEGMENT_LENGTH,
)
import src.preferences

THREAD_QUEUE_COUNT = 2


@pytest.fixture
def db_with_segments(mocker):
    def _create_segment_db(query_return_value=None, length=1):
        if query_return_value is None:
            query_return_value = [mocker.Mock()]

        segment_db = mocker.Mock()
        segment_db.store_segments = mocker.Mock()
        segment_db.query_segment_pairs = mocker.Mock(return_value=query_return_value)
        segment_db.__len__ = mocker.Mock(return_value=length)

        return segment_db

    return _create_segment_db


@pytest.fixture
def feedback_collection_process(
    db_with_segments, mocker, queue_with_items, trajectory_queue_with_items
):
    def _create_feedback_collection_process(
        segment_db=db_with_segments(),
        segment_queue=None,
        preference_queue=None,
        trajectory_queue=None,
        reward_modelling_queue=None,
    ):
        if trajectory_queue is None:
            trajectory_queue = trajectory_queue_with_items([])

        if segment_queue is None:
            segment_queue = queue_with_items([])

        if preference_queue is None:
            preference_queue = queue_with_items([])

        if reward_modelling_queue is None:
            reward_modelling_queue = queue_with_items([])

        mocker.patch(
            "src.preferences.SegmentDB",
            return_value=segment_db,
        )
        mocker.patch("src.preferences.PreferenceElicitationThread.run")

        thread_queue_count = 0

        def create_queue():
            nonlocal thread_queue_count
            assert (
                thread_queue_count < THREAD_QUEUE_COUNT
            ), f"Should only create {THREAD_QUEUE_COUNT} thread queues."
            if thread_queue_count == 0:
                thread_queue_count += 1
                return segment_queue
            if thread_queue_count == 1:
                thread_queue_count += 1
                return preference_queue

        mocker.patch("src.preferences.ThreadQueue", side_effect=create_queue)
        mocker.patch(
            "src.preferences.ProcessQueue", return_value=reward_modelling_queue
        )

        feedback_collection = FeedbackCollectionProcess(
            trajectory_queue=trajectory_queue
        )

        return feedback_collection

    return _create_feedback_collection_process


@pytest.fixture
def queue_with_items(mocker):
    def _create_queue(items: List):
        def empty():
            return len(items) == 0

        def get():
            assert (
                len(items) > 0
            ), "Can only call get on Queue if the there are items in the Queue!"
            item = items.pop()
            return item

        queue = mocker.Mock()
        queue.empty = mocker.Mock(side_effect=empty)
        queue.get = mocker.Mock(side_effect=get)
        queue.put = mocker.Mock()

        return queue

    return _create_queue


@pytest.fixture
def trajectory_queue_with_items(queue_with_items):
    def _create_queue(items: List):
        return queue_with_items(["END"] + items)

    return _create_queue


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

    feedback_collection = feedback_collection_process(trajectory_queue=trajectory_queue)
    feedback_collection.run()

    assert feedback_collection.trajectory_queue.get.call_count == 4


def test_feedback_collection_process_run_initializes_segment_db(
    feedback_collection_process,
):
    feedback_collection = feedback_collection_process()
    feedback_collection.run()

    assert src.preferences.SegmentDB.call_count == 1


def test_feedback_collection_process_run_starts_preference_elicitation_thread(
    feedback_collection_process, mocker
):
    preference_queue = mocker.Mock()

    feedback_collection = feedback_collection_process(preference_queue=preference_queue)
    feedback_collection.run()

    assert src.preferences.PreferenceElicitationThread.run.call_count == 1
    src.preferences.PreferenceElicitationThread.run.assert_called_with(
        queue=preference_queue
    )


def test_feedback_collection_process_stores_segments_from_available_trajectories(
    db_with_segments, feedback_collection_process, mocker, trajectory_queue_with_items
):
    segment_db = db_with_segments()
    trajectory = [(mocker.Mock(), mocker.Mock()) for _ in range(SEGMENT_LENGTH * 3)]
    segment1 = trajectory[:SEGMENT_LENGTH]
    segment2 = trajectory[SEGMENT_LENGTH : SEGMENT_LENGTH * 2]
    segment3 = trajectory[SEGMENT_LENGTH * 2 :]
    trajectory_queue = trajectory_queue_with_items([trajectory])

    feedback_collection = feedback_collection_process(
        segment_db=segment_db, trajectory_queue=trajectory_queue
    )
    feedback_collection.run()

    assert segment_db.store_segments.call_count == 1
    segment_db.store_segments.assert_called_with([segment1, segment2, segment3])


def test_feedback_collection_process_sends_segments_to_preference_elicitor(
    db_with_segments,
    feedback_collection_process,
    mocker,
    queue_with_items,
    trajectory_queue_with_items,
):
    trajectory = [(mocker.Mock(), mocker.Mock()) for _ in range(SEGMENT_LENGTH * 2)]
    segment1 = trajectory[:SEGMENT_LENGTH]
    segment2 = trajectory[SEGMENT_LENGTH : SEGMENT_LENGTH * 2]

    segment_db = db_with_segments(query_return_value=[(segment1, segment2)])
    segment_queue = queue_with_items([])

    trajectory_queue = trajectory_queue_with_items([trajectory])

    feedback_collection = feedback_collection_process(
        trajectory_queue=trajectory_queue,
        segment_queue=segment_queue,
        segment_db=segment_db,
    )
    feedback_collection.run()

    assert segment_db.query_segment_pairs.call_count == 1
    assert segment_queue.put.call_count == 1
    segment_queue.put.assert_called_with((segment1, segment2))


def test_feedback_collection_process_sends_preferences_to_reward_modeller(
    feedback_collection_process, mocker, queue_with_items, trajectory_queue_with_items
):
    trajectory = [(mocker.Mock(), mocker.Mock()) for _ in range(SEGMENT_LENGTH * 2)]
    reward_modelling_queue = queue_with_items([])
    preference = mocker.Mock()
    preference_queue = queue_with_items([preference])
    trajectory_queue = trajectory_queue_with_items([trajectory])

    feedback_collection = feedback_collection_process(
        preference_queue=preference_queue,
        trajectory_queue=trajectory_queue,
        reward_modelling_queue=reward_modelling_queue,
    )
    feedback_collection.run()

    assert reward_modelling_queue.put.call_count == 1
    reward_modelling_queue.put.assert_called_with(preference)
