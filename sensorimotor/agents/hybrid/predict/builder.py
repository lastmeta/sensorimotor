import math
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


class PredictionModel:
    def __init__(self, input_dim: int, output_dim: int):
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
        # self.mse_threshold = 0.01
        # self.model.compile(optimizer='adam', loss='mse')
        # for online learning...
        optimizer = SGD(lr=0.01)
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer, 
            metrics=['accuracy'])

    @staticmethod
    def dim_layer(dim: int, multiplier: int = 6):
        return 2 ** math.ceil(math.log2(multiplier * dim))

    @staticmethod
    def calculate_mse(predictions, actual):
        return np.mean((predictions - actual) ** 2)

    @staticmethod
    def calculate_variance(prediction: list):
        ''' calculates the variance of a given the output of the model '''
        return np.var(prediction, axis=0)

    def train(self, x_train, y_train, epochs=None, batch_size=None):
        ''' incremental training'''
        if not self.model:
            return
        if epochs:
            self.model.fit(
                x_train, y_train,
                epochs=epochs,
                batch_size=1 if x_train.shape[0] == 1 else batch_size)
            return
        for iters in range(1000):
            self.model.fit(x_train, y_train, epochs=1000)
            predictions = self.model.predict(x_train)
            if PredictionModel.calculate_mse(predictions, y_train) <= self.mse_threshold:
                print(f"Mapping learned after {iters+1} epochs.")
                break


comment = '''
Certainly, adaptive models are designed to update themselves as new data comes in. Here are two examples, one from neural networks and one from decision trees:

1. Neural Network: Online Learning with Stochastic Gradient Descent
Neural networks trained with stochastic gradient descent (SGD) can be adapted for online learning. The idea is to update the model weights incrementally, after each observation or small batch of observations.

Here is a very simple neural network using Keras with TensorFlow backend that can be used for online learning:

python
Copy code
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# Assuming a problem with 10 features and a single output
model = Sequential()
model.add(Dense(5, input_dim=10, activation='relu'))  # Input layer and 1 hidden layer
model.add(Dense(1, activation='sigmoid'))  # Output layer

optimizer = SGD(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Then, for online learning, you would call fit() for each new piece of data
# x_new and y_new are the new pieces of data coming in
model.fit(x_new, y_new, batch_size=1, epochs=1)
With batch_size=1 and epochs=1, the model's fit method updates the weights after each new observation.

2. Decision Tree: Hoeffding Trees (Very Fast Decision Trees - VFDT)
Hoeffding Trees or VFDTs are decision trees designed for learning from data streams where it's infeasible to store the entire dataset in memory, and the data distribution can change over time. These trees make use of the Hoeffding bound to decide on the sufficient number of observations before making a split.

scikit-multiflow is a Python library that contains an implementation of the Hoeffding Tree:

python
Copy code
from skmultiflow.trees import HoeffdingTreeClassifier

# Initialize the Hoeffding Tree
ht = HoeffdingTreeClassifier()

# To train incrementally, we use 'partial_fit' method
# x_new and y_new are the new pieces of data
ht.partial_fit(x_new, y_new)
Is XGBoost Incremental?
XGBoost does have an option for incremental learning, which it refers to as "continued training." You can continue training with new data by using the xgb.train method and setting the parameter xgb_model to the previously trained model. However, XGBoost isn't typically categorized as an online learning algorithm since it was designed for batch learning, and its incremental learning capabilities are limited compared to models like Hoeffding Trees.

Remember, these models can serve as starting points, but depending on your specific problem and data distribution, they might require additional adjustments or considerations for features like drift handling, concept change, and forgetting mechanisms.
'''
