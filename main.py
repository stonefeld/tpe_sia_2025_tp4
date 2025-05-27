import argparse

from src.hopfield.hopfield_main import run_hopfield
from src.kohonen.kohonen_main import run_kohonen, run_kohonen_hexagonal
from src.oja.oja_main import run_oja


def main():
    arg_parser = argparse.ArgumentParser(description="Redes Neuronales y Aprendizaje No Supervisado")
    subparsers = arg_parser.add_subparsers(dest="algorithm", required=True)
    subparsers.add_parser("hopfield", help="Correr la red Hopfield")
    subparsers.add_parser("kohonen", help="Correr la red Kohonen")
    subparsers.add_parser("kohonen-hexagonal", help="Correr la red Kohonen en una malla hexagonal")
    subparsers.add_parser("oja", help="Correr la regla de aprendizaje de Oja")
    args = arg_parser.parse_args()

    if args.algorithm == "hopfield":
        print("Ejecutando la red Hopfield...")
        run_hopfield()
    elif args.algorithm == "kohonen":
        print("Ejecutando la red Kohonen...")
        run_kohonen()
    elif args.algorithm == "kohonen-hexagonal":
        print("Ejecutando la red Kohonen en una malla hexagonal...")
        run_kohonen_hexagonal()
    elif args.algorithm == "oja":
        print("Ejecutando la regla de aprendizaje de Oja...")
        run_oja()


if __name__ == "__main__":
    main()
