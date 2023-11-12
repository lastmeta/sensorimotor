import numpy as np
from sklearn.exceptions import NotFittedError
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
        self.model: xgb.XGBClassifier = None
        self.model_type: ModelModel = model_type
        self.initialize_model()

    def initialize_model(self, y_train=None):
        # Determine model type based on y_train or self.model_type
        # if self.model_type is None:
        #    if y_train is not None and len(np.unique(y_train)) > 2:
        #        self.model_type = 'regression'
        #    else:
        #        self.model_type = 'classification'
        # if self.model_type == 'classification':
        #    self.model = xgb.XGBClassifier(
        #        eval_metric='logloss',
        #        objective='binary:logistic' if len(
        #            np.unique(y_train)) == 2 else 'multi:softprob')
        # elif self.model_type == 'regression':
        #    self.model = xgb.XGBRegressor(
        #        eval_metric='rmse',
        #        objective='reg:squarederror')
        # else:
        #    raise ValueError(
        #        "Invalid model type. Must be 'regression' or 'classification'.")
        self.model: xgb.XGBClassifier = xgb.XGBClassifier(
            eval_metric='logloss',
            objective='binary:logistic' if len(
                np.unique(y_train)) == 2 else 'multi:softprob')

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
        print('training xgboost model...')
        print('x_train.shape:', x_train.shape)
        print('y_train.shape:', y_train.shape)
        print('x_train:', x_train)
        print('y_train:', y_train)
        self.model.fit(x_train, y_train, verbose=True)

    def predict(self, x, round=True):
        ''' Make predictions with the trained model '''
        try:
            prediction = self.model.predict(x)
            ret = []
            for p in prediction:
                if isinstance(p, float) and round:
                    ret.append(round(p))
                else:
                    ret.append(p)
            return ret
        except NotFittedError:
            return None
        except Exception as e:
            print('error:', e)
            return None
