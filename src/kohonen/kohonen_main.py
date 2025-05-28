from typing import Dict

from src.kohonen.kohonen_network import KohonenHexagonal, KohonenSquare
from src.kohonen.kohonen_plots import plot_hexagonal_som_assignments, plot_hexagonal_som_distance_map, plot_square_som_assignments, plot_square_som_distance_map, plot_square_som_country_counts_heatmap, plot_square_som_variable_heatmap
from src.utils import load_countries_data


def run_kohonen(init_opts: Dict, train_opts: Dict):
    entities, data = load_countries_data("assets/europe.csv")
    shape = init_opts.pop("shape", "square")

    if shape == "square":
        som = KohonenSquare(entities, data, **init_opts)
        history = som.train(**train_opts)
        plot_square_som_assignments(som)
        plot_square_som_distance_map(som)
        plot_square_som_country_counts_heatmap(history, entities, som.k)
        plot_square_som_variable_heatmap(som, variable_index=1, variable_name="GDP")
        plot_square_som_variable_heatmap(som, variable_index=2, variable_name="Inflation")

    elif shape == "hexagonal":
        som = KohonenHexagonal(entities, data, **init_opts)
        som.train(**train_opts)
        plot_hexagonal_som_assignments(som)
        plot_hexagonal_som_distance_map(som)
