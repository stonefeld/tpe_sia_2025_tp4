import argparse
import json
from typing import Any, Dict

from src.hopfield.hopfield_main import run_hopfield
from src.kohonen.kohonen_main import run_kohonen
from src.oja.oja_main import run_oja


def run_algorithm(config: Dict[str, Any]) -> None:
    algorithm = config.get("algorithm")
    init_opts = config.get("init_opts", {})
    train_opts = config.get("train_opts", {})

    if algorithm == "kohonen":
        print("Ejecutando la red Kohonen...")
        run_kohonen(init_opts, train_opts)
    elif algorithm == "oja":
        print("Ejecutando la regla de aprendizaje de Oja...")
        run_oja(init_opts, train_opts)
    elif algorithm == "hopfield":
        print("Ejecutando la red Hopfield...")
        run_hopfield(init_opts, train_opts)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def main():
    arg_parser = argparse.ArgumentParser(description="Redes Neuronales y Aprendizaje No Supervisado")
    arg_parser.add_argument("config_files", nargs="+", help="JSON configuration files specifying algorithms and options")
    args = arg_parser.parse_args()

    for config_file in args.config_files:
        try:
            with open(config_file, "r") as f:
                config = json.load(f)

            print(f"\nProcessing configuration from {config_file}")
            run_algorithm(config)

        except FileNotFoundError:
            print(f"Error: Configuration file {config_file} not found")

        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in configuration file {config_file}")

        except Exception as e:
            print(f"Error processing {config_file}: {str(e)}")


if __name__ == "__main__":
    main()
