import numpy as np

class Oja:
    def __init__(self, entities, data, learning_rate=0.01, standarization='zscore'):
        """
        Inicializa el modelo de Oja.
        :param entities: lista de nombres (por ejemplo, países)
        :param data: matriz de datos (numérica)
        :param learning_rate: tasa de aprendizaje
        :param standarization: 'zscore' o None
        """
        self.entities = entities
        self.learning_rate = learning_rate

        if standarization == "zscore":
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            data = (data - mean) / std

        self.data = data
        self.input_dim = data.shape[1]
        self.weights = np.random.rand(self.input_dim)

    def train(self, epochs=500):
        """
        Entrena la red neuronal usando la regla de Oja.
        :param epochs: cantidad de épocas de entrenamiento
        :return: vector de pesos normalizado (primera componente principal)
        """
        for _ in range(epochs):
            for x in self.data:
                y = np.dot(self.weights, x)
                self.weights += self.learning_rate * y * (x - y * self.weights)

        self.weights /= np.linalg.norm(self.weights)
        return self.weights

    def project(self):
        """
        Proyecta los datos sobre la componente principal obtenida.
        :return: array con las proyecciones
        """
        return self.data @ self.weights
