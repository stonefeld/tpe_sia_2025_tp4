from typing import Dict

import numpy as np
from sklearn.decomposition import PCA

from src.oja.oja_network import Oja
from src.oja.oja_plots import plot_projection, plot_projection_difference, plot_scatter_oja_vs_pca
from src.utils import load_countries_data, standarize


def run_oja(init_opts: Dict, train_opts: Dict):
    # === Cargar datos ===
    entities, data = load_countries_data("assets/europe.csv")

    # === Entrenar modelo de Oja ===
    oja = Oja(entities, data, **init_opts)
    w_oja = oja.train(**train_opts)
    print(f"Componente principal (Regla de Oja): {w_oja.tolist()}")

    # === Comparar con PCA (sklearn) ===
    pca = PCA(n_components=1)
    data_zscore = standarize(data, "zscore")
    pca.fit(data_zscore)
    w_pca = pca.components_[0]
    print(f"Componente principal (PCA sklearn): {w_pca.tolist()}")

    # === Comparar ángulo entre vectores
    cos_theta = np.dot(w_oja, w_pca) / (np.linalg.norm(w_oja) * np.linalg.norm(w_pca))
    angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    print(f"Ángulo entre vectores (Oja vs PCA): {angle_deg:.2f}°")

    proj_oja = oja.project()
    proj_pca = data_zscore @ w_pca

    plot_projection(data_zscore, entities, w_oja, title="Proyección sobre PC1 (Oja)", save_path="results/oja_proyeccion.png")
    plot_projection(data_zscore, entities, w_pca, title="Proyección sobre PC1 (PCA)", save_path="results/pca_proyeccion.png")
    plot_projection_difference(entities, proj_oja, proj_pca, save_path="results/diff_oja_pca.png")
    plot_scatter_oja_vs_pca(proj_oja, proj_pca, save_path="results/scatter_oja_pca.png")
