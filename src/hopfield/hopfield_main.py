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
        "X":np.array([
            [ 1, -1, -1, -1,  1],
            [-1,  1, -1,  1, -1],
            [-1, -1,  1, -1, -1],
            [-1,  1, -1,  1, -1],
            [ 1, -1, -1, -1,  1]
        ]),
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
    original = patterns[-1]
    noisy = original.copy()
    flip_indices = np.random.choice(len(noisy), size=5, replace=False)
    noisy[flip_indices] *= -1

    # Recuperación
    history = net.recall(noisy, steps=5)

    # Visualización
    plot_comparison(original, noisy, history[-1])
    plot_recall_steps(history)

    # ESTADO ESPURIO
    random_pattern = np.random.choice([-1, 1], size=25)
    history = net.recall(random_pattern, steps=5)
    plot_recall_steps(history, filepath="results/hopfield_espurios.png")
