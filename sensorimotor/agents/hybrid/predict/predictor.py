'''
Predictor models - for every action we can take we build 2 models:
    1.  'next' A model that predicts the next state given the current state
    2.  'prior' A model that predicts the previous state given the current state
We continually train these models as we explore the environment, and use them to
guide our pathfinding.
'''
import numpy as np
from sensorimotor.agents.hybrid.predict.builder import PredictionModel


class Predictor:
    def __init__(self, actions: list, dimension: int):
        self.dimension: int = dimension
        self.actions: list = []
        self.futures: dict = {}
        self.pasts: dict = {}
        self.build_models(actions)

    def build_models(self, actions: list):
        for x in actions:
            self.add_action(x)

    def add_action(self, action):
        self.actions.append(action)
        self.futures[action] = PredictionModel(
            input_dim=self.dimension,
            output_dim=self.dimension)
        self.pasts[action] = PredictionModel(
            input_dim=self.dimension,
            output_dim=self.dimension)

    @staticmethod
    def validateAction(func):
        def inner(self, action, *args, **kwargs):
            if action not in self.actions:
                raise ValueError(f"Action '{action}' not found.")
            return func(self, action, *args, **kwargs)
        return inner

    @validateAction
    def train_future_action(self, action, x_train, y_train):
        self.futures.get(action).train(x_train, y_train)

    @validateAction
    def train_past_action(self, action, x_train, y_train):
        self.pasts.get(action).train(x_train, y_train)

    @validateAction
    def predict_future(self, state, action):
        return self.futures.get(action).model.predict(state)

    @validateAction
    def predict_past(self, state, action):
        return self.pasts.get(action).model.predict(state)

    @validateAction
    def predict_future_uncertainty(self, state, action):
        return PredictionModel.calculate_variance(
            self.predict_future(state, action))

    def actions_by_expected_information_gain(self, state, unexplored_actions):
        '''
        the metric we use is the variance of the prediction of the next state
        which might be only loosely correlated with the the model's confidence
        which is what we really want to use as a metric. If this is insufficient
        we will have to modify the model in order to capture confidence:
        https://chat.openai.com/share/7d7c12f6-7880-4083-8a59-16e64ca85f5e see
        "To extract an uncertainty measure..."
        '''

        def order_actions_by_variance(action_variance: dict) -> list:
            ''' orders the actions by their variance, lowest to highest '''
            ordered_actions = sorted(action_variance, key=action_variance.get)
            return ordered_actions

        return order_actions_by_variance(
            action_variance={
                action: self.predict_future_uncertainty(state, action)
                for action in unexplored_actions
                if action in self.actions})
