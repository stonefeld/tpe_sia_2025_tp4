import numpy as np

from src.utils import get_decay_fn, standarize


class Oja:
    def __init__(self, entities, x, **kwargs):
        self.entities = entities
        self.x = standarize(x, kwargs.get("standarization", "zscore"))
        self.learning_rate = kwargs.get("learning_rate", 0.1)
        self.weights = np.random.uniform(0, 1, x.shape[1])
        self.decay_fn = get_decay_fn(kwargs.get("decay_fn", "inverse"))

    def train(self, **kwargs):
        epochs = kwargs.get("epochs", 1000)

        for epoch in range(epochs):
            lr = self.decay_fn(self.learning_rate, epoch, epochs)

            for xi in self.x:
                y = np.inner(xi, self.weights)
                self.weights += lr * y * (xi - y * self.weights)

        return self.weights

    def project(self):
        return self.x @ self.weights
