import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.oja import Oja
from src.oja_plots import plot_projection, plot_projection_difference, plot_scatter_oja_vs_pca


def load_data(filepath):
    df = pd.read_csv(filepath)
    entities = df["Country"].tolist()
    data = df.drop("Country", axis=1).values
    return entities, data


def main():
    # === Cargar datos ===
    entities, data = load_data("assets/europe.csv")

    # === Entrenar modelo de Oja ===
    oja = Oja(entities, data, learning_rate=0.01, standarization="zscore")
    w_oja = oja.train(epochs=500)
    print("Componente principal (Regla de Oja):")
    print(w_oja)

    # === Comparar con PCA (sklearn) ===
    pca = PCA(n_components=1)
    data_zscore = (data - data.mean(axis=0)) / data.std(axis=0)  # mismo tipo de normalización
    pca.fit(data_zscore)
    w_pca = pca.components_[0]
    print("\nComponente principal (PCA sklearn):")
    print(w_pca)

    # === Comparar ángulo entre vectores
    cos_theta = np.dot(w_oja, w_pca) / (np.linalg.norm(w_oja) * np.linalg.norm(w_pca))
    angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    print(f"\nÁngulo entre vectores (Oja vs PCA): {angle_deg:.2f}°")
    # Proyecciones
    proj_oja = oja.project()
    proj_pca = data_zscore @ w_pca

    # Gráficos
    plot_projection(data, entities, w_oja, title="Proyección sobre PC1 (Oja)", save_path="results/oja_projection.png")
    plot_projection(data_zscore, entities, w_pca, title="Proyección sobre PC1 (PCA)", save_path="results/pca_projection.png")
    plot_projection_difference(entities, proj_oja, proj_pca, save_path="results/diff_oja_pca.png")
    plot_scatter_oja_vs_pca(proj_oja, proj_pca, save_path="results/scatter_oja_pca.png")


if __name__ == "__main__":
    main()
