from enum import Enum
from abc import ABC, abstractmethod


class ModelType(Enum):
    XGBoost = 1
    NeuralNet = 2

    def description(self):
        if self == ModelType.XGBoost:
            return "xgboost model."
        elif self == ModelType.NeuralNet:
            return "a neural net which requires tensorflow."


class ModelModel(ABC):
    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = None

    @abstractmethod
    def train(self, x_train, y_train, epochs=1, batch_size=None):
        ''' Incremental training. Must be implemented in subclasses. '''
        pass

    @abstractmethod
    def predict(self, x):
        ''' Predicts. Must be implemented in subclasses. '''
        pass
