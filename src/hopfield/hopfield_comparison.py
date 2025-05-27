import matplotlib.pyplot as plt
import numpy as np

from src.hopfield.hopfield_network import Hopfield


def letter_patterns(n):
    # Define una letra básica para tamaño n x n
    p = np.ones((n, n))
    p[1:-1, 1:-1] = -1
    return [p.flatten(), np.flipud(p).flatten()]


def compare_hopfield():
    ns = [5, 7, 9]
    flip_bits = [1, 3, 5, 10]
    results = []
    for n in ns:
        patterns = letter_patterns(n)
        net = Hopfield(size=n * n)
        net.train(np.array(patterns))
        orig = patterns[0]
        recover_rates = []
        for flips in flip_bits:
            correct = 0
            for _ in range(20):
                noisy = orig.copy()
                idx = np.random.choice(len(noisy), flips, replace=False)
                noisy[idx] *= -1
                recall = net.recall(noisy, steps=10)[-1]
                if np.all(recall == orig):
                    correct += 1
            recover_rates.append(correct / 20)
        results.append(recover_rates)

    for i, n in enumerate(ns):
        plt.plot(flip_bits, results[i], marker="o", label=f"{n}x{n}")
    plt.xlabel("Bits alterados")
    plt.ylabel("Tasa de recuperación")
    plt.title("Robustez vs tamaño de red y ruido")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    compare_hopfield()
