import math
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class PredictionModel:
    def __init__(self, input_dim: int, output_dim: int):
        self.mse_threshold = 0.01
        self.model = Sequential([
            Dense(
                PredictionModel.dim_layer(input_dim, multiplier=6),
                input_dim=input_dim,
                activation='relu'),
            Dense(
                PredictionModel.dim_layer(output_dim, multiplier=3),
                activation='relu'),
            Dense(output_dim, activation='linear')
        ])
        self.model.compile(optimizer='adam', loss='mse')

    @staticmethod
    def dim_layer(dim: int, multiplier: int = 6):
        return 2 ** math.ceil(math.log2(multiplier * dim))

    @staticmethod
    def calculate_mse(predictions, actual):
        return np.mean((predictions - actual) ** 2)

    def train(self, x_train, y_train, epochs=None):
        if not self.model:
            return
        if epochs:
            self.model.fit(x_train, y_train, epochs=epochs)
            return
        for iters in range(1000):
            self.model.fit(x_train, y_train, epochs=1000)
            predictions = self.model.predict(x_train)
            if PredictionModel.calculate_mse(predictions, y_train) <= self.mse_threshold:
                print(f"Mapping learned after {iters+1} epochs.")
                break
