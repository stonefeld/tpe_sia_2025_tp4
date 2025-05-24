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
        linecolor="black",
        ax=ax,
    )
    ax.set_title("Cantidad de Países por Neurona")
    ax.set_xlabel("")
    ax.set_ylabel("")

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
