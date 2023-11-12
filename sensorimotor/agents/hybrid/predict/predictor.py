'''
Predictor models - for every action we can take we build 2 models:
    1.  'next' A model that predicts the next state given the current state
    2.  'prior' A model that predicts the previous state given the current state
We continually train these models as we explore the environment, and use them to
guide our pathfinding.
'''
from sensorimotor.agents.hybrid.predict.builder import ModelType
from sensorimotor.agents.hybrid.predict.builder.nn import NeuralNetModel
from sensorimotor.agents.hybrid.predict.builder.xgb import XgboostModel


class Predictor:
    def __init__(self, actions: list, dimension: int, model_type: ModelType = None):
        if model_type is None or model_type == ModelType.XGBoost:
            self.model = XgboostModel()
        self.model_type = model_type
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
        if self.model_type is None or self.model_type == ModelType.XGBoost:
            self.futures[action] = XgboostModel(model_type='regression')
            self.pasts[action] = XgboostModel(model_type='regression')
        elif self.model_type == ModelType.NeuralNet:
            self.futures[action] = NeuralNetModel(
                input_dim=self.dimension,
                output_dim=self.dimension)
            self.pasts[action] = NeuralNetModel(
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
    def predict_future(self, action, state):
        return self.futures.get(action).predict(state)

    @validateAction
    def predict_past(self, action, state):
        return self.pasts.get(action).predict(state)

    @validateAction
    def predict_future_uncertainty(self, action, state):
        future = self.predict_future(action, state)
        if future is None or len(future) == 0:
            return None
        if self.model_type == ModelType.XGBoost:
            return XgboostModel.calculate_variance(future)
        elif self.model_type == ModelType.NeuralNet:
            return NeuralNetModel.calculate_variance(future)

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
            try:
                ordered_actions = sorted(
                    action_variance, key=action_variance.get)
                return ordered_actions
            except Exception as e:
                return []

        return order_actions_by_variance(
            action_variance={
                action: self.predict_future_uncertainty(action, state)
                for action in unexplored_actions
                if action in self.actions})
