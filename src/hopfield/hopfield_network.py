import numpy as np


class Hopfield:
    def __init__(self, **kwargs):
        self.size = kwargs.get("size", 5)
        self.weights = np.zeros((self.size, self.size))

    def train(self, patterns, **kwargs):
        for p in patterns:
            p = p.reshape(self.size, 1)
            self.weights += np.dot(p, p.T)

        self.weights /= self.size
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, **kwargs):
        state = pattern.copy()
        history = [(state.copy(), self.energy(state))]
        n = self.size
        steps = kwargs.get("steps", 50)

        for _ in range(steps):
            indices = np.random.permutation(n)
            changed = False

            for i in indices:
                s = np.dot(self.weights[i], state)
                new_value = 1 if s >= 0 else -1

                if new_value != state[i]:
                    state[i] = new_value
                    changed = True

            history.append((state.copy(), self.energy(state)))

            if not changed:
                break

        return history

    def energy(self, state):
        return -0.5 * np.dot(state.T, np.dot(self.weights, state))
