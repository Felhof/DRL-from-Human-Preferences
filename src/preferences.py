from dataclasses import dataclass
import itertools
from multiprocessing import Process, Queue
from queue import Queue as ThreadQueue
from threading import Thread
from typing import List, Tuple

import cv2
import numpy as np
from src.utils import Trajectory

SEGMENT_LENGTH = 300
CLIP_BORDER_HEIGHT = 84
CLIP_BORDER_WIDTH = 10


@dataclass
class Preference:
    segment1: "Segment"
    segment2: "Segment"
    mu: float


class Segment:
    def __init__(self: "Segment", data: Trajectory) -> None:
        self.data = data

    def get_observations(self: "Segment") -> List[np.ndarray]:
        return [p[0] for p in self.data]


class SegmentDB:
    def __init__(self: "SegmentDB") -> None:
        self.segments: List[Segment] = []
        self.queried_pairs = []

    def __len__(self: "SegmentDB") -> int:
        return len(self.segments)

    def store_segments(self: "SegmentDB", new_segments: List[Segment]) -> None:
        self.segments.extend(new_segments)

    def query_segment_pair(self: "SegmentDB", n: int = 1) -> Tuple[Segment, Segment]:

        all_indices = list(range(len(self.segments)))
        possible_pair_indices = list(itertools.combinations(all_indices, 2))

        assert len(possible_pair_indices) > len(
            self.queried_pairs
        ), "Not enough segments left to query."

        np.random.shuffle(possible_pair_indices)

        for possible_pair_index in possible_pair_indices:
            if possible_pair_index in self.queried_pairs:
                continue
            self.queried_pairs.append(possible_pair_index)
            return (
                self.segments[possible_pair_index[0]],
                self.segments[possible_pair_index[1]],
            )


def ask_for_evaluation(p_queue: ThreadQueue) -> None:
    p = ""
    while p not in ["E", "I", "L", "R"]:
        p = input(
            "Please indicate a preference for the left (L) or right (R) clip by typing L or R or indicate "
            "indifference by typing E. If you consider the clips incomparable, type I."
        )
    p_queue.put(p)


class FeedbackCollectionProcess(Process):
    def __init__(
        self: "FeedbackCollectionProcess",
        preference_queue: Queue,
        trajectory_queue: Queue,
        stop_queue: Queue,
    ) -> None:
        super().__init__()
        self.preference_queue = preference_queue
        self.trajectory_queue = trajectory_queue
        self.stop_queue = stop_queue
        self.segment_db = None

    def _update_segment_db(
        self: "FeedbackCollectionProcess", trajectory: Trajectory
    ) -> None:
        segments: List[Segment] = [
            Segment(trajectory[i : i + SEGMENT_LENGTH])
            for i in range(0, len(trajectory), SEGMENT_LENGTH)
        ]
        self.segment_db.store_segments(segments)

    def get_preference_from_segment_pair(
        self: "FeedbackCollectionProcess", segment_pair: Tuple[Segment, Segment]
    ) -> str:
        border = np.zeros((CLIP_BORDER_HEIGHT, CLIP_BORDER_WIDTH), dtype=np.uint8)

        clip1 = segment_pair[0].get_observations()
        clip2 = segment_pair[1].get_observations()

        p_queue = ThreadQueue()

        evaluation_thread = Thread(target=ask_for_evaluation, args=(p_queue,))
        evaluation_thread.start()

        cv2.namedWindow("ClipWindow")
        while evaluation_thread.is_alive():
            for clip1_frame, clip2_frame in zip(clip1, clip2):
                frame = np.hstack((clip1_frame, border, clip2_frame))
                cv2.imshow("ClipWindow", frame)
                cv2.waitKey(25)
        cv2.destroyWindow("ClipWindow")

        preference = p_queue.get()

        return preference

    def run(self: "FeedbackCollectionProcess"):
        self.segment_db = SegmentDB()

        while True:
            if not self.stop_queue.empty():
                if self.stop_queue.get():
                    break

            if not self.trajectory_queue.empty():
                trajectory = self.trajectory_queue.get()
                self._update_segment_db(trajectory)

            if len(self.segment_db) > 0:
                segment_pair = self.segment_db.query_segment_pair()
                preference = self.get_preference_from_segment_pair(segment_pair)
                if preference == "I":
                    continue
                mu = {"L": 0.0, "R": 1.0, "E": 0.5}[preference]
                self.preference_queue.put(
                    Preference(
                        segment1=segment_pair[0], segment2=segment_pair[1], mu=mu
                    )
                )
