import matplotlib.pyplot as plt
import numpy as np

from src.kohonen.kohonen_network import KohonenSquare
from src.utils import load_countries_data, standarize


def compare_kohonen():
    entities, data = load_countries_data("assets/europe.csv")
    methods = ["zscore", "unit", "minmax"]
    lrs = [0.05, 0.1, 0.2]
    weight_inits = ["uniform", "sample"]
    rs = [1, 2, 3]
    decay_fns = [
        ("Constante", lambda x, t, m: x),
        ("1/(1+t)", lambda x, t, m: x / (1 + t)),
        ("Exponencial", lambda x, t, m: x * np.exp(-t / m)),
    ]

    fig, axes = plt.subplots(len(methods), len(lrs), figsize=(15, 10), sharex=True, sharey=True)
    for i, std in enumerate(methods):
        for j, lr in enumerate(lrs):
            som = KohonenSquare(
                entities,
                data,
                k=4,
                r=2,
                learning_rate=lr,
                standarization=std,
                weight_init="sample",
                decay_fn=lambda x, t, m: x * np.exp(-t / m),
            )
            som.train(epochs=1000)
            mapping = som.map_input()
            hits = np.zeros((4, 4))
            for _, idx in mapping:
                hits[idx // 4, idx % 4] += 1
            ax = axes[i, j]
            im = ax.imshow(hits, cmap="Purples", vmin=0)
            ax.set_title(f"std={std}, lr={lr}")
            ax.axis("off")
    plt.suptitle("Comparación: Método de estandarización y Learning Rate")
    plt.tight_layout()
    plt.show()

    # Inicialización y decay
    fig, axes = plt.subplots(len(weight_inits), len(decay_fns), figsize=(15, 7))
    for i, wi in enumerate(weight_inits):
        for j, (df_name, df) in enumerate(decay_fns):
            som = KohonenSquare(
                entities,
                data,
                k=4,
                r=2,
                learning_rate=0.1,
                standarization="zscore",
                weight_init=wi,
                decay_fn=df,
            )
            som.train(epochs=1000)
            mapping = som.map_input()
            hits = np.zeros((4, 4))
            for _, idx in mapping:
                hits[idx // 4, idx % 4] += 1
            ax = axes[i, j]
            im = ax.imshow(hits, cmap="Purples", vmin=0)
            ax.set_title(f"init={wi}, decay={df_name}")
            ax.axis("off")
    plt.suptitle("Comparación: Inicialización de pesos y Función de decay")
    plt.tight_layout()
    plt.show()

    # Comparación de r
    fig, axes = plt.subplots(1, len(rs), figsize=(15, 3))
    for j, r in enumerate(rs):
        som = KohonenSquare(
            entities,
            data,
            k=4,
            r=r,
            learning_rate=0.1,
            standarization="zscore",
            weight_init="sample",
            decay_fn=lambda x, t, m: x * np.exp(-t / m),
        )
        som.train(epochs=1000)
        mapping = som.map_input()
        hits = np.zeros((4, 4))
        for _, idx in mapping:
            hits[idx // 4, idx % 4] += 1
        ax = axes[j]
        im = ax.imshow(hits, cmap="Purples", vmin=0)
        ax.set_title(f"r={r}")
        ax.axis("off")
    plt.suptitle("Comparación: r")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_kohonen()
