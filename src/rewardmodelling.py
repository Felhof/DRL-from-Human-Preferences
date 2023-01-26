from typing import List

import numpy as np

from src.preferences import Preference

BUFFER_SIZE = 3000


class PreferenceBuffer:
    def __init__(self: "PreferenceBuffer") -> None:
        self.preferences: List[Preference] = []
        self.number_of_preferences = 0
        self.idx = 0

    def __len__(self: "PreferenceBuffer") -> int:
        return self.number_of_preferences

    def add(self, preference: Preference) -> None:
        if self.number_of_preferences < BUFFER_SIZE:
            self.preferences.append(preference)
            self.number_of_preferences += 1
        else:
            self.preferences[self.idx] = preference
        self.idx = (self.idx + 1) % BUFFER_SIZE

    def get_minibatch(self, n=32) -> List[Preference]:
        indices = list(range(0, self.number_of_preferences))
        minibatch_indices = np.random.choice(indices, size=n, replace=False)

        return [self.preferences[i] for i in minibatch_indices]
