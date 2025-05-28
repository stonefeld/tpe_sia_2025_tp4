from abc import ABC, abstractmethod

import numpy as np

from src.utils import get_decay_fn, standarize


class Kohonen(ABC):
    def __init__(self, entities, x, **kwargs):
        self.entities = entities
        self.x = standarize(x, kwargs.get("standarization", "zscore"))
        self.k = kwargs.get("k", 5)
        self.learning_rate = kwargs.get("learning_rate", 0.1)
        self.r = kwargs.get("r", self.k)

        n = self.x.shape[1]
        neurons = self.k**2
        weight_init = kwargs.get("weight_init", "uniform")

        if weight_init == "uniform":
            self.weights = [np.random.uniform(0, 1, n) for _ in range(neurons)]
        elif weight_init == "sample":
            idxs = np.random.choice(len(self.x), neurons, replace=True)
            self.weights = [np.array(self.x[i]) for i in idxs]
        else:
            raise ValueError(f"Unknown weight_init method: {weight_init}")

        self.decay_fn = get_decay_fn(kwargs.get("decay_fn", "exponential"))

    def train(self, **kwargs):
        method = kwargs.get("train_method", "batch")
        epochs = kwargs.get("epochs", 1000)
        train_fn = self._get_train_fn(method)

        for epoch in range(epochs):
            # Reducimos la tasa de aprendizaje y el radio
            lr = self.decay_fn(self.learning_rate, epoch, epochs)
            r = self.decay_fn(self.r, epoch, epochs)

            # Entrenamos
            train_fn(lr, r)

    def map_entities(self):
        return [(e, self._get_winner(x)) for e, x in zip(self.entities, self.x)]

    # GETTERS
    def _get_winner(self, xi):
        return np.argmin([self._euclidean_distance(w, xi) for w in self.weights])

    def _get_neighbors(self, winner_idx, r):
        winner_coords = np.array(divmod(winner_idx, self.k))
        neighbors = []

        for i in range(self.k**2):
            coords = np.array(divmod(i, self.k))
            if self._calculate_distance(winner_coords, coords) <= r:
                neighbors.append(i)

        return neighbors

    # TRAINING METHODS
    def _get_train_fn(self, method):
        if method == "batch":
            return self._train_batch
        elif method == "stochastic":
            return self._train_stochastic
        else:
            raise ValueError(f"Unknown training method: {method}")

    def _train_batch(self, lr, r):
        for xi in self.x:
            winner = self._get_winner(xi)
            self._update_weights(xi, winner, r)

    def _train_stochastic(self, lr, r):
        xi = self.x[np.random.choice(len(self.x))]
        winner = self._get_winner(xi)
        self._update_weights(xi, winner, r)

    def _update_weights(self, xi, winner_idx, r):
        neighbors = self._get_neighbors(winner_idx, r)
        for i in neighbors:
            self.weights[i] += self.learning_rate * (xi - self.weights[i])

    # DISTANCE METHODS
    def _euclidean_distance(self, a, b):
        # return np.sqrt(np.sum((a - b) ** 2))
        return np.linalg.norm(a - b)

    @abstractmethod
    def _calculate_distance(self, a, b):
        pass


class KohonenSquare(Kohonen):
    def _calculate_distance(self, a, b):
        return self._euclidean_distance(a, b)


class KohonenHexagonal(Kohonen):
    def _calculate_distance(self, a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy, (dx + dy) // 2)
