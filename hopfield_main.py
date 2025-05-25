import numpy as np

from src.hopfield import Hopfield
from src.hopfield_plots import plot_comparison, plot_recall_steps


def main():
    # Letras A, E, H, J con formas claras
    letters = {
        "A": np.array(
            [
                [-1, 1, 1, 1, -1],
                [1, -1, -1, -1, 1],
                [1, 1, 1, 1, 1],
                [1, -1, -1, -1, 1],
                [1, -1, -1, -1, 1],
            ]
        ),
        "E": np.array(
            [
                [1, 1, 1, 1, 1],
                [1, -1, -1, -1, -1],
                [1, 1, 1, -1, -1],
                [1, -1, -1, -1, -1],
                [1, 1, 1, 1, 1],
            ]
        ),
        "H": np.array(
            [
                [1, -1, -1, -1, 1],
                [1, -1, -1, -1, 1],
                [1, 1, 1, 1, 1],
                [1, -1, -1, -1, 1],
                [1, -1, -1, -1, 1],
            ]
        ),
        "J": np.array(
            [
                [1, 1, 1, 1, 1],
                [-1, -1, -1, 1, -1],
                [-1, -1, -1, 1, -1],
                [1, -1, -1, 1, -1],
                [1, 1, 1, -1, -1],
            ]
        ),
    }

    # Entrenamiento
    patterns = [v.flatten() for v in letters.values()]
    net = Hopfield(size=patterns[0].size)
    net.train(np.array(patterns))

    # Selección de letra y ruido
    original = patterns[-1]
    noisy = original.copy()
    flip_indices = np.random.choice(len(noisy), size=5, replace=False)
    noisy[flip_indices] *= -1

    # Recuperación
    history = net.recall(noisy, steps=5)

    # Visualización
    plot_comparison(original, noisy, history[-1])
    plot_recall_steps(history)


if __name__ == "__main__":
    main()
