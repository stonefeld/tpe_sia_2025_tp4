import math

from src.kohonen.kohonen_network import KohonenHexagonal, KohonenSquare
from src.kohonen.kohonen_plots import plot_hexagonal_som_assignments, plot_hexagonal_som_distance_map, plot_square_som_assignments, plot_square_som_distance_map
from src.utils import load_countries_data


def run_kohonen():
    entities, data = load_countries_data("assets/europe.csv")
    som = KohonenSquare(
        entities,
        data,
        k=5,
        # r=2,
        learning_rate=0.1,
        standarization="zscore",
        weight_init="sample",
        # decay_fn=lambda x, t, m: x * 1 / (1 + t),
        decay_fn=lambda x, t, m: x * math.exp(-t / m),
    )
    som.train(epochs=2000)

    plot_square_som_assignments(som)
    plot_square_som_distance_map(som)


def run_kohonen_hexagonal():
    entities, data = load_countries_data("assets/europe.csv")
    som = KohonenHexagonal(
        entities,
        data,
        k=4,
        r=2,
        learning_rate=0.1,
        standarization="zscore",
        weight_init="sample",
        # decay_fn=lambda x, t, m: x * 1 / (1 + t),
        decay_fn=lambda x, t, m: x * math.exp(-t / m),
    )
    som.train(epochs=1000)

    plot_hexagonal_som_assignments(som)
    plot_hexagonal_som_distance_map(som)


if __name__ == "__main__":
    run_kohonen_hexagonal()
