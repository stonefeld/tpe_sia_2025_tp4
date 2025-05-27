import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils import save_plot


# SQUARE PLOTS
def plot_square_som_assignments(som):
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
        cmap="Purples",
        cbar_kws={"label": "Cantidad de países"},
        ax=ax,
    )
    ax.set_title("Cantidad de Países por Neurona")

    cbar = ax.collections[0].colorbar
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(0.5)

    # Ponemos las etiquestas de los países en el centro de cada celda
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


def plot_square_som_distance_map(som):
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

            w = weights[i, j]
            umatrix[i, j] = np.mean([np.sqrt(np.sum((w - n) ** 2)) for n in neighbors])

    umatrix = (umatrix - np.min(umatrix)) / (np.max(umatrix) - np.min(umatrix))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        umatrix,
        annot=True,
        cmap="YlOrRd",
        cbar_kws={"label": "Distancia promedio"},
        ax=ax,
    )

    ax.set_title("Distancias promedio entre neuronas vecinas (U-Matrix)")

    cbar = ax.collections[0].colorbar
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(0.5)

    save_plot(fig, "results/kohonen_umatrix.png")
    plt.show()


# HEXAGONAL PLOTS
def plot_hexagonal_heatmap(data, title, cmap="Purples", cbar_label="Value", annotate=True):
    k = data.shape[0]
    fig, ax = plt.subplots(figsize=(10, 8))

    # Hexagon geometry
    hex_radius = 1.0
    dx = 3 / 2 * hex_radius
    dy = np.sqrt(3) * hex_radius / 2

    # Calculate hexagon centers
    centers = []
    for row in range(k):
        for col in range(k):
            x = col * dx
            y = row * 2 * dy + (col % 2) * dy
            centers.append((x, y))

    # Draw hexagons
    for idx, (x, y) in enumerate(centers):
        row = idx // k
        col = idx % k
        value = data[row, col]
        color = plt.cm.get_cmap(cmap)(value / data.max() if data.max() > 0 else 0)
        # Hexagon vertices
        hexagon = plt.Polygon(
            [
                (
                    x + hex_radius * np.cos(np.pi / 3 * i),
                    y + hex_radius * np.sin(np.pi / 3 * i),
                )
                for i in range(6)
            ],
            closed=True,
            fill=True,
            color=color,
        )
        ax.add_patch(hexagon)
        if annotate:
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", color="white" if value > data.max() / 2 else "black", fontsize=10)

    ax.set_xlim(-hex_radius, dx * (k - 1) + hex_radius * 2)
    ax.set_ylim(-hex_radius, 2 * dy * (k - 1) + 2 * dy + hex_radius)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, data.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(cbar_label)

    return fig, ax


def plot_hexagonal_som_assignments(som):
    results = som.map_input()
    heatmap = np.zeros((som.k, som.k), dtype=int)
    entity_map = [[[] for _ in range(som.k)] for _ in range(som.k)]

    for e, i in results:
        row, col = i // som.k, i % som.k
        heatmap[row, col] += 1
        entity_map[row][col].append(e)

    fig, ax = plot_hexagonal_heatmap(heatmap, "Cantidad de Países por Neurona (Hexagonal)", cmap="Purples", cbar_label="Cantidad de países", annotate=False)

    # Add country names at correct hex centers
    k = som.k
    hex_radius = 1.0
    dx = 3 / 2 * hex_radius
    dy = np.sqrt(3) * hex_radius / 2
    for row in range(k):
        for col in range(k):
            names = entity_map[row][col]
            if names:
                x = col * dx
                y = row * 2 * dy + (col % 2) * dy
                text = "\n".join(names)
                value = heatmap[row, col]
                norm_value = value / heatmap.max() if heatmap.max() > 0 else 0
                color = "white" if norm_value > 0.5 else "black"
                ax.text(x, y, text, ha="center", va="center", color=color, clip_on=True, fontsize=10)

    save_plot(fig, "results/kohonen_hexagonal_mapa_asignaciones.png")
    plt.show()


def plot_hexagonal_som_distance_map(som):
    weights = np.array(som.weights).reshape(som.k, som.k, -1)
    umatrix = np.zeros((som.k, som.k))
    k = som.k

    # Hexagonal neighbor offsets (even-q vertical layout)
    neighbor_offsets = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]
    neighbor_offsets_odd = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
    for row in range(k):
        for col in range(k):
            neighbors = []
            offsets = neighbor_offsets if col % 2 == 0 else neighbor_offsets_odd
            for dr, dc in offsets:
                nr, nc = row + dr, col + dc
                if 0 <= nr < k and 0 <= nc < k:
                    neighbors.append(weights[nr, nc])

            w = weights[row, col]
            umatrix[row, col] = np.mean([np.sqrt(np.sum((w - n) ** 2)) for n in neighbors])

    umatrix = (umatrix - np.min(umatrix)) / (np.max(umatrix) - np.min(umatrix))
    fig, ax = plot_hexagonal_heatmap(
        umatrix,
        "Distancias promedio entre neuronas vecinas (U-Matrix Hexagonal)",
        cmap="YlOrRd",
        cbar_label="Distancia promedio",
        annotate=True,
    )

    save_plot(fig, "results/kohonen_hexagonal_umatrix.png")
    plt.show()
