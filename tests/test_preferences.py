import builtins
import threading

import cv2
import pytest
from typing import List

from src.preferences import (
    FeedbackCollectionProcess,
    Preference,
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
        segment_db.query_segment_pair = mocker.Mock(return_value=query_return_value)
        segment_db.__len__ = mocker.Mock(return_value=length)

        return segment_db

    return _create_segment_db


@pytest.fixture
def feedback_collection_process(
    db_with_segments, mocker, queue_with_items, stop_queue_with_items
):
    def _create_feedback_collection_process(
        segment_db=db_with_segments(),
        trajectory_queue=None,
        preference_queue=None,
        mock_preferences=None,
        evaluation_thread=None,
        stop_queue=None,
    ):

        if trajectory_queue is None:
            trajectory_queue = queue_with_items([])

        if preference_queue is None:
            preference_queue = queue_with_items([])

        if stop_queue is None:
            stop_queue = stop_queue_with_items([])

        mocker.patch(
            "src.preferences.SegmentDB",
            return_value=segment_db,
        )
        mocker.patch("src.preferences.cv2.namedWindow")
        mocker.patch("src.preferences.cv2.imshow")
        mocker.patch("src.preferences.cv2.waitKey")
        mocker.patch("src.preferences.cv2.destroyWindow")
        mocker.patch("src.preferences.Thread", return_value=evaluation_thread)

        feedback_collection = FeedbackCollectionProcess(
            preference_queue=preference_queue,
            trajectory_queue=trajectory_queue,
            stop_queue=stop_queue,
        )

        if mock_preferences is not None:
            feedback_collection.get_preference_from_segment_pair = mocker.Mock(
                side_effect=mock_preferences
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
def stop_queue_with_items(queue_with_items):
    def _create_queue(items: List):
        return queue_with_items([True] + items)

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

    mocker.patch(
        "src.preferences.itertools.combinations", return_value=[(0, 2), (1, 3)]
    )

    segment_pair1 = segment_db.query_segment_pair()
    segment_pair2 = segment_db.query_segment_pair()

    assert segment_pair1 != segment_pair2
    assert segment_pair1 == (segment1, segment3) or (segment2, segment4)
    assert segment_pair2 == (segment1, segment3) or (segment2, segment4)
    assert all(
        [queried_pair in [(1, 3), (0, 2)] for queried_pair in segment_db.queried_pairs]
    )


def test_feedback_collection_process_run_runs_until_receiving_end_message(
    feedback_collection_process, mocker, queue_with_items, stop_queue_with_items
):
    stop_queue = stop_queue_with_items([False, False, False])
    trajectory_queue = queue_with_items(["A", "B", "C", "D"])

    feedback_collection = feedback_collection_process(
        stop_queue=stop_queue,
        trajectory_queue=trajectory_queue,
        mock_preferences=mocker.Mock(return_value="I"),
    )
    feedback_collection.run()

    assert feedback_collection.trajectory_queue.get.call_count == 3
    assert feedback_collection.stop_queue.get.call_count == 4


def test_feedback_collection_process_run_initializes_segment_db(
    feedback_collection_process,
):
    feedback_collection = feedback_collection_process()
    feedback_collection.run()

    assert src.preferences.SegmentDB.call_count == 1


def test_feedback_collection_process_stores_segments_from_available_trajectories(
    db_with_segments,
    feedback_collection_process,
    mocker,
    stop_queue_with_items,
    queue_with_items,
):
    segment_db = db_with_segments()
    trajectory = [(mocker.Mock(), mocker.Mock()) for _ in range(SEGMENT_LENGTH * 3)]
    segment1 = trajectory[:SEGMENT_LENGTH]
    segment2 = trajectory[SEGMENT_LENGTH : SEGMENT_LENGTH * 2]
    segment3 = trajectory[SEGMENT_LENGTH * 2 :]
    trajectory_queue = queue_with_items([trajectory])
    stop_queue = stop_queue_with_items([False])

    feedback_collection = feedback_collection_process(
        segment_db=segment_db,
        trajectory_queue=trajectory_queue,
        stop_queue=stop_queue,
        mock_preferences=mocker.Mock(return_value="I"),
    )
    feedback_collection.run()

    assert segment_db.store_segments.call_count == 1
    segment_list = segment_db.store_segments.call_args.args[0]
    assert segment_list[0].data == segment1
    assert segment_list[1].data == segment2
    assert segment_list[2].data == segment3


def test_feedback_collection_process_asks_for_preference_from_preference_elicitor(
    db_with_segments,
    feedback_collection_process,
    mocker,
    queue_with_items,
    stop_queue_with_items,
):
    trajectory = [(mocker.Mock(), mocker.Mock()) for _ in range(SEGMENT_LENGTH * 2)]
    segment1 = trajectory[:SEGMENT_LENGTH]
    segment2 = trajectory[SEGMENT_LENGTH : SEGMENT_LENGTH * 2]

    segment_db = db_with_segments(query_return_value=(segment1, segment2))

    trajectory_queue = queue_with_items([trajectory])
    stop_queue = stop_queue_with_items([False])

    feedback_collection = feedback_collection_process(
        trajectory_queue=trajectory_queue,
        segment_db=segment_db,
        stop_queue=stop_queue,
        mock_preferences=mocker.Mock(return_value="I"),
    )
    feedback_collection.run()

    assert segment_db.query_segment_pair.call_count == 1
    assert feedback_collection.get_preference_from_segment_pair.call_count == 1
    feedback_collection.get_preference_from_segment_pair.assert_called_with(
        (segment1, segment2)
    )


@pytest.mark.parametrize(
    "preference, expected_mu", [("L", 0.0), ("R", 1.0), ("E", 0.5)]
)
def test_feedback_collection_process_sends_preferences_to_reward_modeller(
    preference,
    expected_mu,
    db_with_segments,
    feedback_collection_process,
    mocker,
    queue_with_items,
    stop_queue_with_items,
):
    trajectory = [(mocker.Mock(), mocker.Mock()) for _ in range(SEGMENT_LENGTH * 2)]
    segment1 = Segment(trajectory[:SEGMENT_LENGTH])
    segment2 = Segment(trajectory[SEGMENT_LENGTH : SEGMENT_LENGTH * 2])

    preference_queue = queue_with_items([])
    segment_db = db_with_segments(query_return_value=(segment1, segment2))

    def return_preference_for_segment_pair(segment_pair):
        if segment_pair == (segment1, segment2):
            return preference
        return mocker.Mock()

    trajectory_queue = queue_with_items([trajectory])
    stop_queue = stop_queue_with_items([False])

    feedback_collection = feedback_collection_process(
        trajectory_queue=trajectory_queue,
        preference_queue=preference_queue,
        segment_db=segment_db,
        mock_preferences=return_preference_for_segment_pair,
        stop_queue=stop_queue,
    )
    feedback_collection.run()

    assert preference_queue.put.call_count == 1
    preference_queue.put.assert_called_with(
        Preference(
            segment1=segment1,
            segment2=segment2,
            mu=expected_mu,
        ),
    )


def test_feedback_collection_process_does_not_send_preference_for_incomparable_segments(
    db_with_segments,
    feedback_collection_process,
    mocker,
    queue_with_items,
    stop_queue_with_items,
):
    trajectory = [(mocker.Mock(), mocker.Mock()) for _ in range(SEGMENT_LENGTH * 2)]
    segment1 = Segment(trajectory[:SEGMENT_LENGTH])
    segment2 = Segment(trajectory[SEGMENT_LENGTH : SEGMENT_LENGTH * 2])

    preference_queue = queue_with_items([])
    segment_db = db_with_segments(query_return_value=(segment1, segment2))

    def return_preference_for_segment_pair(segment_pair):
        if segment_pair == (segment1, segment2):
            return "I"
        return mocker.Mock()

    trajectory_queue = queue_with_items([trajectory])
    stop_queue = stop_queue_with_items([False])

    feedback_collection = feedback_collection_process(
        trajectory_queue=trajectory_queue,
        preference_queue=preference_queue,
        segment_db=segment_db,
        mock_preferences=return_preference_for_segment_pair,
        stop_queue=stop_queue,
    )
    feedback_collection.run()

    assert preference_queue.put.call_count == 0


def test_feedback_collection_process_can_generate_preference_from_segment_pair(
    feedback_collection_process, mocker
):
    # Given
    segment1 = Segment([(mocker.Mock(), mocker.Mock()) for _ in range(5)])
    segment2 = Segment([(mocker.Mock(), mocker.Mock()) for _ in range(5)])
    segment_pair = (segment1, segment2)
    expected_preference = "R"

    evaluation_thread = mocker.Mock()
    evaluation_thread.start = mocker.Mock()
    evaluation_thread.is_alive = mocker.Mock(side_effect=[True, False])

    feedback_collection = feedback_collection_process(
        evaluation_thread=evaluation_thread
    )

    mocker.patch("src.preferences.np.hstack", return_value=mocker.Mock())
    thread_queue = mocker.Mock()
    thread_queue.get = mocker.Mock(return_value="R")
    mocker.patch("src.preferences.ThreadQueue", return_value=thread_queue)

    # When
    received_preference = feedback_collection.get_preference_from_segment_pair(
        segment_pair
    )

    # Then
    src.preferences.Thread.assert_called_once_with(
        target=src.preferences.ask_for_evaluation, args=(thread_queue,)
    )

    cv2.namedWindow.assert_called_once_with("ClipWindow")
    assert cv2.imshow.call_count == 5
    assert cv2.imshow.call_count == 5
    cv2.destroyWindow.assert_called_once_with("ClipWindow")

    assert evaluation_thread.start.call_count == 1
    assert evaluation_thread.is_alive.call_count == 2

    assert expected_preference == received_preference


def test_can_ask_for_evaluation(mocker):
    thread_queue = mocker.Mock()
    thread_queue.put = mocker.Mock()
    mocker.patch("builtins.input", return_value="R")

    src.preferences.ask_for_evaluation(thread_queue)

    builtins.input.assert_called_once_with(
        "Please indicate a preference for the left (L) or right (R) clip by typing L or R or indicate indifference by "
        "typing E. If you consider the clips incomparable, type I."
    )
    thread_queue.put.assert_called_once_with("R")
