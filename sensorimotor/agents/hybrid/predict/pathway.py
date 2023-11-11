import numpy as np
from sensorimotor.agents.hybrid.predict.builder import ModelType
from sensorimotor.agents.hybrid.predict.builder.nn import NeuralNetModel
from sensorimotor.agents.hybrid.predict.builder.xgb import XgboostModel


class Pathway:
    def __init__(self, dimension: int, model_type: ModelType = None):
        self.dimension = dimension
        self.model_type = model_type
        if model_type is None or model_type == ModelType.XGBoost:
            self.model = XgboostModel()
        elif model_type == ModelType.NeuralNet:
            self.model = NeuralNetModel(
                input_dim=self.dimension * 2,  # current state and goal state
                output_dim=self.dimension)     # middle state

    def train_middle_state(self, current_state, goal_state, middle_state):
        x_train = np.concatenate([current_state, goal_state])
        y_train = np.array([middle_state])
        if self.model is XgboostModel:
            self.model.train(x_train.reshape(1, -1), y_train.reshape(-1))
        elif self.model is NeuralNetModel:
            # """Without this reshaping, you'd be trying to pass a 1D array to the
            # model's train method, which would likely result in an error because
            # the method is expecting a 2D array where the first axis is the batch
            # size (number of samples)."""
            self.model.train(x_train.reshape(1, -1), y_train)

    def predict_middle_state(self, current_state, goal_state):
        x_test = np.concatenate([current_state, goal_state])
        return self.model.predict(x_test.reshape(1, -1))
