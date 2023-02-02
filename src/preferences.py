import logging
import logging.handlers
from dataclasses import dataclass
import itertools
from multiprocessing import Process, Queue
import os
from queue import Queue as ThreadQueue
import sys
from threading import Thread
from typing import List, Tuple, Optional

import cv2
import numpy as np
from src.utils import Trajectory

SEGMENT_LENGTH = 300
CLIP_SIZE = 126
CLIP_BORDER_WIDTH = 30
TRAJECTORY_QUEUE_CAPACITY = 5


@dataclass
class Preference:
    segment1: "Segment"
    segment2: "Segment"
    mu: float


class Segment:
    def __init__(self: "Segment", data: Trajectory) -> None:
        self.data = data

    def __eq__(self: "Segment", other: "Segment") -> bool:
        if len(self.data) != len(other.data):
            return False
        return all(
            [
                np.all(s[0] == o[0]) and np.all(s[0] == o[0])
                for s, o in zip(self.data, other.data)
            ]
        )

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

    def query_segment_pair(self: "SegmentDB") -> Optional[Tuple[Segment, Segment]]:

        all_indices = list(range(len(self.segments)))
        possible_pair_indices = list(itertools.combinations(all_indices, 2))

        np.random.shuffle(possible_pair_indices)

        for possible_pair_index in possible_pair_indices:
            if possible_pair_index in self.queried_pairs:
                continue
            self.queried_pairs.append(possible_pair_index)
            return (
                self.segments[possible_pair_index[0]],
                self.segments[possible_pair_index[1]],
            )
        return None


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

        self.logger = logging.getLogger(self.name)

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
        border = np.zeros((CLIP_SIZE, CLIP_BORDER_WIDTH), dtype=np.uint8)

        clip1 = segment_pair[0].get_observations()
        clip2 = segment_pair[1].get_observations()

        p_queue = ThreadQueue()

        evaluation_thread = Thread(target=ask_for_evaluation, args=(p_queue,))
        evaluation_thread.start()

        cv2.namedWindow("ClipWindow")
        while evaluation_thread.is_alive():
            for framestack1, framestack2 in zip(clip1, clip2):
                clip1_frame = framestack1[-1]
                clip2_frame = framestack2[-1]
                clip1_frame = cv2.resize(
                    clip1_frame, (CLIP_SIZE, CLIP_SIZE), interpolation=cv2.INTER_AREA
                )
                clip2_frame = cv2.resize(
                    clip2_frame, (CLIP_SIZE, CLIP_SIZE), interpolation=cv2.INTER_AREA
                )
                frame = np.hstack((clip1_frame, border, clip2_frame))
                cv2.imshow("ClipWindow", frame)
                cv2.waitKey(50)
        cv2.destroyWindow("ClipWindow")

        preference = p_queue.get()

        return preference

    def run(self: "FeedbackCollectionProcess"):
        self.logger.info("Starting preference collection process.")

        """
        FeedbackCollectionProcess will be a child process and so stdin is automatically closed, resulting in an error when asking for input.
        Workaround courtesy of: https://github.com/mrahtz/learning-from-human-preferences/blob/3fca07c4c3fd20bec307f4405684461437d9e215/run.py#L287
        """
        sys.stdin = os.fdopen(0)

        self.segment_db = SegmentDB()

        while True:
            if not self.stop_queue.empty():
                if self.stop_queue.get():
                    self.logger.info("Received stop signal.")
                    break

            self.logger.info(
                f"Trying to get up to {TRAJECTORY_QUEUE_CAPACITY} trajectories from the trajectory queue."
            )

            for _ in range(TRAJECTORY_QUEUE_CAPACITY):
                if self.trajectory_queue.empty():
                    self.logger.info("No more trajectories in the trajectory queue.")
                    break

                trajectory = self.trajectory_queue.get()
                self.logger.info(
                    "Got a trajectory from the trajectory queue and putting it in the DB."
                )
                self._update_segment_db(trajectory)

            if len(self.segment_db) > 1:
                self.logger.info("Querying segment pair to ask for preference.")
                maybe_segment_pair = self.segment_db.query_segment_pair()
                if maybe_segment_pair is None:
                    self.logger.info("No unqueried segment pair found.")
                    continue
                segment_pair = maybe_segment_pair
                self.logger.info(
                    "Found unqueried segment pair. Asking user for preference."
                )
                preference = self.get_preference_from_segment_pair(segment_pair)
                if preference == "I":
                    self.logger.info("User deemed segment pair incomparable.")
                    continue
                mu = {"L": 0.0, "R": 1.0, "E": 0.5}[preference]
                self.logger.info(
                    f"User expressed preference: {preference}. Putting preference in queue."
                )
                self.preference_queue.put(
                    Preference(
                        segment1=segment_pair[0], segment2=segment_pair[1], mu=mu
                    )
                )
