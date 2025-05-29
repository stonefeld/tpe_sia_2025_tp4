from typing import Dict

import numpy as np

from src.hopfield.hopfield_network import Hopfield
from src.hopfield.hopfield_plots import plot_comparison, plot_recall_steps


def run_hopfield(init_opts: Dict, train_opts: Dict):
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
        "X": np.array(
            [
                [1, -1, -1, -1, 1],
                [-1, 1, -1, 1, -1],
                [-1, -1, 1, -1, -1],
                [-1, 1, -1, 1, -1],
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
    net = Hopfield(size=patterns[0].size, **init_opts)
    net.train(np.array(patterns), **train_opts)

    # Selección de letra y ruido
    train_letter = np.array(train_opts.get("letter", letters["J"]))
    original = train_letter.flatten()
    noisy = original.copy()
    noise = init_opts.get("noise", 0.2)
    noise_indices = np.random.choice(len(noisy), size=int(net.size * noise), replace=False)
    noisy[noise_indices] *= -1

    # Recuperación
    history = net.recall(noisy, steps=5)

    # Visualización
    plot_comparison(original, noisy, history[-1][0])
    plot_recall_steps(history)
