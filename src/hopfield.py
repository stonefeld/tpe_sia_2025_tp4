# hopfield.py
import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for p in patterns:
            p = p.reshape(self.size, 1)
            self.weights += np.dot(p, p.T)
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, steps=10):
        state = pattern.copy()
        history = [state.copy()]
        for _ in range(steps):
            new_state = state.copy()
            for i in range(self.size):
                s = np.dot(self.weights[i], state)
                new_state[i] = 1 if s >= 0 else -1
            state = new_state
            history.append(state.copy())
        return np.array(history)

    def energy(self, state):
        return -0.5 * np.dot(state.T, np.dot(self.weights, state))
