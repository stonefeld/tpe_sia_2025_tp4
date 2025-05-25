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
    som = Kohonen(entities, data, k=4, learning_rate=0.1, standarization="zscore", weight_init="uniform")
    som.train(epochs=10000, tolerance=1e-4)
    plot_som_assignments(som)
    plot_som_distance_map(som)


if __name__ == "__main__":
    main()
