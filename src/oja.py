import numpy as np

from src.utils import standarize


class Oja:
    def __init__(self, entities, x, learning_rate=0.01, standarization="zscore", decay_fn=None):
        self.entities = entities
        self.learning_rate = learning_rate
        self.x = standarize(x, standarization)
        self.weights = np.random.uniform(0, 1, x.shape[1])
        self.decay_fn = decay_fn if decay_fn else lambda x, t, m: x

    def train(self, epochs=500):
        for epoch in range(epochs):
            lr = self.decay_fn(self.learning_rate, epoch, epochs)

            for xi in self.x:
                y = np.inner(xi, self.weights)
                self.weights += lr * y * (xi - y * self.weights)

        # self.weights /= np.linalg.norm(self.weights)
        return self.weights

    def project(self):
        return self.x @ self.weights
