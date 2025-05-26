import argparse

from src.hopfield.main import run_hopfield
from src.kohonen.main import run_kohonen
from src.oja.main import run_oja


def main():
    arg_parser = argparse.ArgumentParser(description="Redes Neuronales y Aprendizaje No Supervisado")
    subparsers = arg_parser.add_subparsers(dest="algorithm", required=True)
    subparsers.add_parser("hopfield", help="Correr la red Hopfield")
    subparsers.add_parser("kohonen", help="Correr la red Kohonen")
    subparsers.add_parser("oja", help="Correr la regla de aprendizaje de Oja")
    args = arg_parser.parse_args()

    if args.algorithm == "hopfield":
        print("Ejecutando la red Hopfield...")
        run_hopfield()
    elif args.algorithm == "kohonen":
        print("Ejecutando la red Kohonen...")
        run_kohonen()
    elif args.algorithm == "oja":
        print("Ejecutando la regla de aprendizaje de Oja...")
        run_oja()


if __name__ == "__main__":
    main()
