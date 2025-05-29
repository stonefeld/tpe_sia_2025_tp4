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
    elif method == "minmax":
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        ret = (data - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"Unknown standardization method: {method}")

    return ret


def get_decay_fn(decay_type):
    if decay_type == "exponential":
        return lambda x, t, m: x * np.exp(-t / m)
    elif decay_type == "inverse":
        return lambda x, t, m: x * 1 / (1 + t)
    elif decay_type == "linear":
        return lambda x, t, m: x
    else:
        raise ValueError(f"Unknown decay type: {decay_type}")
