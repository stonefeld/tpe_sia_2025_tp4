from abc import ABC, abstractmethod

import numpy as np

from ..utils import standarize


class Kohonen(ABC):
    def __init__(self, entities, x, k=3, learning_rate=0.1, standarization="unit", weight_init="uniform", r=None, decay_fn=None):
        self.entities = entities
        self.x = standarize(x, standarization)
        self.k = k
        self.learning_rate = learning_rate
        self.r = r if r is not None else k

        n = self.x.shape[1]
        neurons = self.k**2

        if weight_init == "uniform":
            self.weights = [np.random.uniform(0, 1, n) for _ in range(neurons)]
        elif weight_init == "sample":
            idxs = np.random.choice(len(self.x), neurons, replace=True)
            self.weights = [np.array(self.x[i]) for i in idxs]
        else:
            raise ValueError(f"Unknown weight_init method: {weight_init}")

        self.decay_fn = decay_fn if decay_fn else lambda x, t, m: x

    def train(self, epochs=1000, method="batch"):
        if method == "batch":
            for epoch in range(epochs):
                lr = self.decay_fn(self.learning_rate, epoch, epochs)
                r = max(self.decay_fn(self.r, epoch, epochs), 1)

                for xi in self.x:
                    wk_idx = self._get_winner(xi)
                    for i in self._get_neighbors(wk_idx, r):
                        self.weights[i] += lr * (xi - self.weights[i])

        elif method == "stochastic":
            for epoch in range(epochs):
                lr = self.decay_fn(self.learning_rate, epoch, epochs)
                r = max(self.decay_fn(self.r, epoch, epochs), 1)

                xi_idx = np.random.choice(len(self.x))
                xi = self.x[xi_idx]
                wk_idx = self._get_winner(xi)
                neighbors = self._get_neighbors(wk_idx, r)

                for i in neighbors:
                    self.weights[i] += lr * (xi - self.weights[i])

        else:
            raise ValueError(f"Unknown training method: {method}")

    def map_input(self):
        return [(e, self._get_winner(x)) for e, x in zip(self.entities, self.x)]

    def _get_winner(self, xi):
        return np.argmin([self._calculate_distance(w, xi) for w in self.weights])

    def _get_neighbors(self, winner_idx, r):
        winner_coords = np.array(divmod(winner_idx, self.k))
        neighbors = []

        for i in range(self.k**2):
            coords = np.array(divmod(i, self.k))
            if self._calculate_distance(winner_coords, coords) <= r:
                neighbors.append(i)

        return neighbors

    @abstractmethod
    def _calculate_distance(self, winner_idx, r):
        pass


class KohonenSquare(Kohonen):
    def _calculate_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))


class KohonenHexagonal(Kohonen):
    def _calculate_distance(self, a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy, (dx + dy) // 2)
