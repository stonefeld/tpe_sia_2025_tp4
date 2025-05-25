from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils import save_plot


def plot_som_assignments(som):
    results = som.map_input()
    heatmap = np.zeros((som.k, som.k), dtype=int)
    entity_map = [[[] for _ in range(som.k)] for _ in range(som.k)]

    for e, i in results:
        row, col = i // som.k, i % som.k
        heatmap[row, col] += 1
        entity_map[row][col].append(e)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        heatmap,
        annot=False,
        cmap="Blues",
        cbar_kws={"label": "Cantidad de países"},
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Cantidad de Países por Neurona")
    ax.set_xlabel("")
    ax.set_ylabel("")

    cbar = ax.collections[0].colorbar
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(0.5)

    # Annotate with entity names
    for row in range(som.k):
        for col in range(som.k):
            names = entity_map[row][col]
            if names:
                text = "\n".join(names)
                value = heatmap[row, col]
                norm_value = value / heatmap.max() if heatmap.max() > 0 else 0
                color = "white" if norm_value > 0.5 else "black"
                ax.text(
                    col + 0.5,
                    row + 0.5,
                    text,
                    ha="center",
                    va="center",
                    color=color,
                    clip_on=True,
                )

    save_plot(fig, "results/kohonen_mapa_asignaciones.png")
    plt.show()


def plot_som_distance_map(som):
    # Convertir la lista de pesos a un array NumPy
    weights = np.array(som.weights).reshape(som.k, som.k, -1)
    umatrix = np.zeros((som.k, som.k))

    for i in range(som.k):
        for j in range(som.k):
            neighbors = []
            if i > 0:
                neighbors.append(weights[i - 1, j])
            if i < som.k - 1:
                neighbors.append(weights[i + 1, j])
            if j > 0:
                neighbors.append(weights[i, j - 1])
            if j < som.k - 1:
                neighbors.append(weights[i, j + 1])
            dists = [np.linalg.norm(weights[i, j] - n) for n in neighbors]
            umatrix[i, j] = np.mean(dists)

    fig, ax = plt.subplots()
    im = ax.imshow(umatrix, cmap="viridis")
    ax.set_title("Distancias promedio entre neuronas vecinas (U-Matrix)")
    fig.colorbar(im, ax=ax)
    save_plot(fig, "results/kohonen_umatrix.png")
    plt.show()
