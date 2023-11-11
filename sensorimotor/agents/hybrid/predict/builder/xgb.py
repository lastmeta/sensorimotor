import numpy as np
import xgboost as xgb
from sensorimotor.agents.hybrid.predict.builder import ModelModel


class XgboostModel(ModelModel):
    '''
    alternative to the neural network model. 
    xgboost is more efficient for small environments or computers without a GPU.
    '''

    def __init__(self, model_type='classification'):
        super().__init__(0, 0)
        # Initialize without a specific model
        self.model = None
        self.model_type = model_type

    def initialize_model(self, y_train=None):
        # Determine model type based on y_train or self.model_type
        if self.model_type is None and y_train is not None:
            self.model_type = 'classification' if len(
                np.unique(y_train)) > 2 else 'regression'
        if self.model_type == 'classification':
            self.model = xgb.XGBClassifier(objective='binary:logistic' if len(
                np.unique(y_train)) == 2 else 'multi:softprob')
        elif self.model_type == 'regression':
            self.model = xgb.XGBRegressor(objective='reg:squarederror')
        else:
            raise ValueError(
                "Invalid model type. Must be 'regression' or 'classification'.")

    @staticmethod
    def calculate_mse(predictions, actual):
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        if not isinstance(actual, np.ndarray):
            actual = np.array(actual)
        return np.mean((predictions - actual) ** 2)

    @staticmethod
    def calculate_variance(prediction: list):
        ''' calculates the variance of a given output of the model '''
        return np.var(prediction, axis=0)

    def train(self, x_train, y_train, epochs=1, batch_size=None):
        ''' Train the XGBoost model '''
        if self.model is None:
            self.initialize_model(y_train)
        if self.model is None:
            return
        self.model.fit(x_train, y_train, eval_metric=(
            "logloss" if self.model_type == 'classification' else "rmse"),
            verbose=True,)

    def predict(self, x):
        ''' Make predictions with the trained model '''
        return self.model.predict(x)
