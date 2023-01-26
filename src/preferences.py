from dataclasses import dataclass
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

    def __len__(self: "SegmentDB") -> int:
        return len(self.segments)

    def store_segments(self: "SegmentDB", new_segments: List[Segment]) -> None:
        self.segments.extend(new_segments)

    def query_segment_pairs(
        self: "SegmentDB", n: int = 1
    ) -> List[Tuple[Segment, Segment]]:
        assert (
            len(self.segments) >= 2 * n
        ), "Not enough segments to get that many pairs!"

        all_indices = list(range(len(self.segments)))
        selected_indices = np.random.choice(all_indices, 2 * n, replace=False)
        np.random.shuffle(selected_indices)
        index_pairs = [
            (selected_indices[i], selected_indices[i + 1])
            for i in range(0, len(selected_indices), 2)
        ]
        segment_pairs = [
            (self.segments[index_pair[0]], self.segments[index_pair[1]])
            for index_pair in index_pairs
        ]
        return segment_pairs


def ask_for_evaluation(p_queue: ThreadQueue) -> None:
    p = ""
    while p not in ["E", "I", "L", "R"]:
        p = input(
            "Please indicate a preference for the left (L) or right (R) clip by typing L or R or indicate "
            "indifference by typing E. If you consider the clips incomparable, type I."
        )
    p_queue.put(p)


class FeedbackCollectionProcess(Process):
    def __init__(self: "FeedbackCollectionProcess", trajectory_queue: Queue) -> None:
        super().__init__()
        self.trajectory_queue: Queue = trajectory_queue
        self.reward_modelling_queue = None
        self.segment_db = None
        self.preference_elicitor = None

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
        self.reward_modelling_queue = Queue()

        while True:
            if not self.trajectory_queue.empty():
                msg = self.trajectory_queue.get()
                if isinstance(msg, str) and msg == "END":
                    break
                self._update_segment_db(msg)

            if len(self.segment_db) > 0:
                segment_pair = self.segment_db.query_segment_pairs()[0]
                preference = self.get_preference_from_segment_pair(segment_pair)
                if preference == "I":
                    continue
                mu = {"L": 0.0, "R": 1.0, "E": 0.5}[preference]
                self.reward_modelling_queue.put(
                    Preference(
                        segment1=segment_pair[0], segment2=segment_pair[1], mu=mu
                    )
                )
