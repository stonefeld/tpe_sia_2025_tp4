import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.oja.oja_network import Oja
from src.utils import load_countries_data, standarize

def compare_oja():
    entities, data = load_countries_data("assets/europe.csv")
    methods = ["zscore", "unit", "minmax"]
    lrs = [0.01, 0.05, 0.1]
    weight_inits = [None, "positive", "negative"]
    decay_fns = [
        ("Constante", lambda x, t, m: x),
        ("1/(1+t)", lambda x, t, m: x / (1 + t)),
        ("Exponencial", lambda x, t, m: x * np.exp(-t / m)),
    ]
    df = pd.read_csv("assets/europe.csv")
    varnames = list(df.columns)[1:]  # Excluir la columna "Country"

    # Comparación de estandarización y learning rate
    fig, axes = plt.subplots(len(methods), len(lrs), figsize=(15, 10))
    for i, std in enumerate(methods):
        for j, lr in enumerate(lrs):
            oja = Oja(
                entities, data,
                learning_rate=lr,
                standarization=std,
                decay_fn=lambda x, t, m: x * np.exp(-t / m),
            )
            w = oja.train(epochs=1000)
            ax = axes[i, j]
            ax.bar(range(len(w)), w)
            ax.set_xticks(range(len(w)))
            ax.set_xticklabels(varnames, rotation=45, ha='right')
            ax.set_title(f"std={std}, lr={lr}")
    plt.suptitle("Comparación: Método de estandarización y Learning Rate")
    plt.tight_layout()
    plt.show()

    # Decay y pesos
    fig, axes = plt.subplots(len(weight_inits), len(decay_fns), figsize=(15, 7))
    for i, wi in enumerate(weight_inits):
        for j, (df_name, df) in enumerate(decay_fns):
            w_init = np.random.uniform(0, 1, data.shape[1]) if wi == "positive" else \
                     np.random.uniform(-1, 0, data.shape[1]) if wi == "negative" else None
            oja = Oja(
                entities, data,
                learning_rate=0.1,
                standarization="zscore",
                decay_fn=df,
            )
            if w_init is not None:
                oja.weights = w_init
            w = oja.train(epochs=1000)
            ax = axes[i, j]
            ax.bar(range(len(w)), w)
            ax.set_xticks(range(len(w)))
            ax.set_xticklabels(varnames, rotation=45, ha='right')
            ax.set_title(f"init={wi or 'random'}, decay={df_name}")
    plt.suptitle("Comparación: Inicialización de pesos y Función de decay")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_oja()
