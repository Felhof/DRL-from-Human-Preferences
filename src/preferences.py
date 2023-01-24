from dataclasses import dataclass
from multiprocessing import Process, Queue
from threading import Thread

import numpy as np


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

    def store_segments(self, new_segments):
        self.segments.extend(new_segments)

    def query_segment_pairs(self, n):
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

    def __init__(self):
        super().__init__()
        self.segment_db = None
        self.segment_queue = None
        self.preference_elicitation = None

    def run(self):
        self.segment_db = SegmentDB()
        self.segment_queue = Queue()
        self.preference_elicitation = PreferenceElicitationThread()
        self.preference_elicitation.run(queue=self.segment_queue)


class PreferenceElicitationThread(Thread):

    def run(self, queue=None):
        pass