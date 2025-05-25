import os

import numpy as np
import pandas as pd
from scipy.stats import zscore


def save_plot(fig, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {filepath}")


def load_countries_data(filepath):
    df = pd.read_csv(filepath)
    entities = df["Country"].tolist()
    data = df.drop("Country", axis=1).values
    return entities, data


def standarize(data, method="zscore"):
    ret = data

    if method == "zscore":
        ret = zscore(data, axis=0)
    elif method == "unit":
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms[norms == 0] = 1
        ret /= norms

    return ret
