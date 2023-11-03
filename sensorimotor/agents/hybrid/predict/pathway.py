import numpy as np
from builder import PredictionModel


class PathwayPredictor:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.pathway_model = PredictionModel(
            input_dim=self.dimension * 2,  # current state and goal state
            output_dim=self.dimension)     # middle state

    def train_middle_state(self, current_state, goal_state, middle_state):
        x_train = np.concatenate([current_state, goal_state])
        y_train = np.array([middle_state])
        self.pathway_model.train(x_train.reshape(1, -1), y_train)

    def predict_middle_state(self, current_state, goal_state):
        x_test = np.concatenate([current_state, goal_state])
        return self.pathway_model.model.predict(x_test.reshape(1, -1))
