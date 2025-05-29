<h1 align="center">Sistemas de Inteligencia Artificial</h1>
<h3 align="center">TP4: Aprendizaje no supervisado</h3>
<h4 align="center">Primer cuatrimestre 2025</h4>

# Requisitos

* Python ([versión 3.12.9](https://www.python.org/downloads/release/python-3129/))
* [UV](https://docs.astral.sh/uv/getting-started/installation/)

# Instalando las dependencias

```bash
# Si python 3.12.9 no esta instalado se puede instalar haciendo
uv python install 3.12.9

# Para crear y activar el entorno virtual
uv venv
source .venv/bin/activate  # En Unix
.venv\Scripts\activate     # En Windows

# Para instalar las dependencias
uv sync
```

# Corriendo el proyecto

El proyecto consta de un archivo _frontend_ con el cuál se puede demostrar el
funcionamiento del motor de los motores de aprendizaje no supervisado. Este
archivo `main.py` recibirá como único parámetro un archivo de configuración que
especifica el algoritmo que se quiera ejecutar.

```bash
uv run main.py configs/config.json
```

Todos los archivos de configuración pueden solo especificar el algoritmo a
utilizar, ya que todas las opciones cuentan con defaults. Por los que se podría
únicamente utilizar este archivo:

```jsonc
{
  "algorithm": "kohonen" // o también "oja" o "hopfield"
}
```

## Ejemplo de config para Kohonen

```json
{
  "algorithm": "kohonen",
  "init_opts": {
    "shape": "square",
    "learning_rate": 0.1,
    "k": 5,
    "r": 2,
    "standarization": "zscore",
    "weight_init": "sample",
    "decay_fn": "exponential"
  },
  "train_opts": {
    "epochs": 100,
    "train_method": "batch"
  }
}
```

## Ejemplo de config para OJA

```json
{
  "algorithm": "oja",
  "init_opts": {
    "learning_rate": 0.1,
    "standarization": "zscore",
    "weight_init": "sample",
    "decay_fn": "exponential"
  },
  "train_opts": {
    "epochs": 100
  }
}
```

## Ejemplo de config para Hopfield

```json
{
  "algorithm": "hopfield",
  "init_opts": {
    "size": 5,
    "noise": 0.2
  },
  "train_opts": {
    "letter": [
      [1, 1, 1, 1, 1],
      [1, -1, -1, -1, 1],
      [1, -1, -1, -1, 1],
      [1, -1, -1, -1, 1],
      [1, 1, 1, 1, 1]
    ]
  }
}
```

# Utilizando los motores de aprendizaje

Si se desea utilizar un _frontend_ propio pero utilizando el motor de este
proyecto, se puede hacer lo siguiente para cada caso:

```python
from src.kohonen.kohonen_network import KohonenSquare, KohonenHexagonal
from src.oja.oja_network import Oja
from src.hopfield.hopfield_network import Hopfield

# Para Kohonen
kohonen = KohonenSquare(entities, data, standarization="zscore")
history = kohonen.train(epochs=1000)
winners = kohonen.map_entities()

# Para OJA
oja = Oja(entities, data)
weights = oja.train(epochs=100)
projection = oja.project()

# Para Hopfield
hopfield = Hopfield(size=5)
hopfield.train(patterns)
history = hopfield.recall(pattern, steps=10)
```

# Adicional: archivos de análisis

Se encuentran para este proyecto además archivos de análisis y comparación
dentro de cada módulo que se pueden ejecutar como un módulo de python haciendo
por ejemplo

```bash
uv run -m src.kohonen.kohonen_analysis
```
