import matplotlib.pyplot as plt
import numpy as np

from src.utils import save_plot


def plot_projection(data, countries, w, title="", save_path=None):
    projections = data @ w
    sorted_idx = np.argsort(projections)
    sorted_countries = np.array(countries)[sorted_idx]
    sorted_proj = projections[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sorted_countries, sorted_proj, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Proyección")
    ax.set_ylabel("Países")

    if save_path:
        save_plot(fig, save_path)

    plt.show()


def plot_projection_difference(entities, proj_oja, proj_pca, save_path=None):
    diff = np.abs(proj_oja - proj_pca)
    sorted_idx = np.argsort(-diff)
    sorted_entities = np.array(entities)[sorted_idx]
    sorted_diff = diff[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sorted_entities, sorted_diff, color="tomato")
    ax.set_title("Diferencia Absoluta entre Oja y PCA por País")
    ax.set_xlabel("Diferencia |Oja - PCA|")
    ax.set_ylabel("País")
    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    plt.show()


def plot_scatter_oja_vs_pca(proj_oja, proj_pca, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(proj_pca, proj_oja, color="purple")
    ax.plot([min(proj_pca), max(proj_pca)], [min(proj_pca), max(proj_pca)], linestyle="--", color="gray")
    ax.set_title("Dispersión: Proyección PCA vs Oja")
    ax.set_xlabel("Proyección PCA")
    ax.set_ylabel("Proyección Oja")
    ax.grid(True)
    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    plt.show()
