'''
Predictor models - for every action we can take we build 2 models:
    1.  'next' A model that predicts the next state given the current state
    2.  'prior' A model that predicts the previous state given the current state
We continually train these models as we explore the environment, and use them to
guide our pathfinding.
'''

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
