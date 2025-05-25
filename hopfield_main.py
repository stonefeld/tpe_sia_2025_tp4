import numpy as np
from src.hopfield import HopfieldNetwork
from src.hopfield_plots import plot_comparison, plot_recall_steps

# Letras A, E, H, J con formas claras
letters = {
    "A": np.array([
        [-1,  1,  1,  1, -1],
        [ 1, -1, -1, -1,  1],
        [ 1,  1,  1,  1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1]
    ]),
    "E": np.array([
        [ 1,  1,  1,  1,  1],
        [ 1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1,  1,  1,  1,  1]
    ]),
    "H": np.array([
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1,  1,  1,  1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1]
    ]),
    "J": np.array([
        [ 1,  1,  1,  1,  1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [ 1,  1,  1, -1, -1]
    ])
}

# Entrenamiento
patterns = [v.flatten() for v in letters.values()]
net = HopfieldNetwork(size=25)
net.train(np.array(patterns))

# Selección de letra y ruido
original = patterns[3]  # Letra A
noisy = original.copy()
flip_indices = np.random.choice(len(noisy), size=5, replace=False)
noisy[flip_indices] *= -1

# Recuperación
history = net.recall(noisy, steps=5)

# Visualización
plot_comparison(original, noisy, history[-1])
plot_recall_steps(history)
