from dataclasses import dataclass
from multiprocessing import Process, Queue
from queue import Queue as ThreadQueue
from threading import Thread

import numpy as np

SEGMENT_LENGTH = 300


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

    def _update_segment_db(self, trajectory):
        segments = [
            trajectory[i : i + SEGMENT_LENGTH]
            for i in range(0, len(trajectory), SEGMENT_LENGTH)
        ]
        self.segment_db.store_segments(segments)

    def run(self):
        self.segment_db = SegmentDB()
        self.reward_modelling_queue = Queue()
        self.preference_elicitor = PreferenceElicitor()

        while True:
            if not self.trajectory_queue.empty():
                msg = self.trajectory_queue.get()
                if isinstance(msg, str) and msg == "END":
                    break
                self._update_segment_db(msg)

            if len(self.segment_db) > 0:
                segment_pair = self.segment_db.query_segment_pairs()[0]
                preference = self.preference_elicitor.ask_for_preference(segment_pair)
                self.reward_modelling_queue.put(preference)


class PreferenceElicitor:
    def __init__(self):
        pass

    def run(self, queue=None):
        pass

    def ask_for_preference(self, segment_pair):
        pass
