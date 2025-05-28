from src.kohonen.kohonen_network import KohonenSquare
from src.kohonen.kohonen_plots import (plot_dead_units_comparison,
                                       save_dead_units_result)
from src.utils import load_countries_data

# Experiment settings
k_range = range(2, 10)  # k values to test
init_methods = ["uniform", "sample"]  # weight initialization methods
csv_file = "dead_units_results.csv"
train_opts = {"epochs": 1000, "train_method": "stochastic"}
runs = 5  # Number of repetitions for averaging


def run_dead_units_experiment():
    entities, data = load_countries_data("assets/europe.csv")
    for k in k_range:
        for method in init_methods:
            for run in range(runs):
                som = KohonenSquare(entities, data, k=k, weight_init=method)
                som.train(**train_opts)
                dead_units = som.count_dead_units()
                save_dead_units_result(csv_file, k, method, dead_units)
                print(f"run={run+1}, k={k}, method={method}, dead_units={dead_units}")
    plot_dead_units_comparison(csv_file)


if __name__ == "__main__":
    run_dead_units_experiment()
