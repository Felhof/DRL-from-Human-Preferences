from dataclasses import dataclass
from multiprocessing import Process, Queue
from queue import Queue as ThreadQueue
from threading import Thread
from typing import Tuple

import cv2
import numpy as np

SEGMENT_LENGTH = 300
CLIP_BORDER_HEIGHT = 84
CLIP_BORDER_WIDTH = 10


@dataclass
class Preference:
    segment1: "Segment"
    segment2: "Segment"
    mu: float


class Segment:
    def __init__(self, data) -> None:
        self.data = data

    def get_observations(self):
        return [p[0] for p in self.data]


class SegmentDB:
    def __init__(self) -> None:
        self.segments = []

    def __len__(self) -> int:
        return len(self.segments)

    def store_segments(self, new_segments):
        self.segments.extend(new_segments)

    def query_segment_pairs(self, n=1):
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


class FeedbackCollectionProcess(Process):
    def __init__(self, trajectory_queue: Queue):
        super().__init__()
        self.trajectory_queue: Queue = trajectory_queue
        self.reward_modelling_queue = None
        self.segment_db = None
        self.preference_elicitor = None

    def _ask_for_evaluation(self):
        pass

    def _update_segment_db(self, trajectory):
        segments = [
            trajectory[i: i + SEGMENT_LENGTH]
            for i in range(0, len(trajectory), SEGMENT_LENGTH)
        ]
        self.segment_db.store_segments(segments)

    def get_preference_from_segment_pair(self, segment_pair: Tuple[Segment, Segment]):
        border = np.zeros((CLIP_BORDER_HEIGHT, CLIP_BORDER_WIDTH), dtype=np.uint8)

        clip1 = segment_pair[0].get_observations()
        clip2 = segment_pair[1].get_observations()

        p_queue = ThreadQueue()

        evaluation_thread = Thread(target=self._ask_for_evaluation, args=(p_queue,))
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

    def run(self):
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
                mu = {
                    "L": 0.,
                    "R": 1.,
                    "E": 0.5
                }[preference]
                self.reward_modelling_queue.put(
                    Preference(
                        segment1=segment_pair[0],
                        segment2=segment_pair[1],
                        mu=mu
                    )
                )
