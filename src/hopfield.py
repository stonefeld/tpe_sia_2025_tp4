import numpy as np


class Hopfield:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for p in patterns:
            p = p.reshape(self.size, 1)
            self.weights += np.dot(p, p.T)

        self.weights /= self.size
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, steps=50):
        state = pattern.copy()
        history = [state.copy()]
        n = self.size

        for _ in range(steps):
            indices = np.random.permutation(n)
            changed = False

            for i in indices:
                s = np.dot(self.weights[i], state)
                new_value = 1 if s >= 0 else -1

                if new_value != state[i]:
                    state[i] = new_value
                    changed = True

            history.append(state.copy())

            if not changed:
                break

        return np.array(history)

    def energy(self, state):
        return -0.5 * np.dot(state.T, np.dot(self.weights, state))
