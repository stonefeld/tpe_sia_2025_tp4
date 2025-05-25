import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_projection(data, countries, w, title="Proyección sobre PC1 (Regla de Oja)", save_path=None):
    """
    Proyecta los datos sobre el vector w y grafica un barchart ordenado.
    """
    projections = data @ w
    sorted_idx = np.argsort(projections)
    sorted_countries = np.array(countries)[sorted_idx]
    sorted_proj = projections[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sorted_countries, sorted_proj, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Proyección")
    ax.set_ylabel("Países")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
