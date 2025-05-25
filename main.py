import math

import pandas as pd

from src.kohonen import Kohonen
from src.plots import plot_som_assignments, plot_som_distance_map


def load_csv(filepath):
    df = pd.read_csv(filepath)
    entities = df["Country"].tolist()
    data = df.drop("Country", axis=1).values
    return entities, data


def main():
    entities, data = load_csv("assets/europe.csv")
    som = Kohonen(
        entities,
        data,
        k=4,
        r=2,
        learning_rate=0.1,
        standarization="zscore",
        weight_init="sample",
        decay_fn=lambda x, t, m: x * 1 / (1 + t),
    )
    som.train(epochs=1000)
    plot_som_assignments(som)
    plot_som_distance_map(som)


if __name__ == "__main__":
    main()
