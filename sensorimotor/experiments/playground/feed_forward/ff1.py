'''
forward both ways
'''
import numpy as np


def normal_CDF(x):
    ''' approximates normal CDF, almost as easy to compute as sigmoid '''
    return 1 / (1 + np.exp(-x/0.55))


def normal_CDF_derivative(x):
    exp_term = np.exp(-x/0.55)
    return exp_term / (0.55 * ((1 + exp_term)**2))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rng = np.random.default_rng(seed)
        # seed middle neurons with random values
        self.i_hidden_output = self.rng.uniform(
            low=0.0, high=1.0, size=(input_size))
        self.o_hidden_output = self.rng.uniform(
            low=0.0, high=1.0, size=(output_size))
        # seed weights with random values
        self.i_hidden_weights = self.rng.uniform(
            low=0.0, high=1.0, size=(input_size, hidden_size))
        self.o_hidden_weights = self.rng.uniform(
            low=0.0, high=1.0, size=(output_size, hidden_size))

        # # notice we update the same weights in both directions instead of
        # # having separate weights for each direction, but this requires that
        # # all the layers be the same size... so we'll have to do something
        # # about that...
        # self.h_output_weights = self.rng.uniform(
        #    low=0.0, high=1.0, size=(hidden_size, output_size))
        # self.h_input_weights = self.rng.uniform(
        #    low=0.0, high=1.0, size=(hidden_size, input_size))

    def train(self, X, y, learning_rate=0.5):
        # from middle to input
        self.h_input_activation = np.dot(
            self.i_hidden_output,
            self.i_hidden_weights)  # self.h_input_weights
        self.h_input_output = normal_CDF(self.h_input_activation)

        # from middle to output
        self.h_output_activation = np.dot(
            self.o_hidden_output,
            self.o_hidden_weights)  # self.h_output_weights
        self.h_output_output = normal_CDF(self.h_output_activation)

        # update i_hidden_weights and o_hidden_weights based on others' output
        self.h_input_errors = X - self.h_input_output
        self.h_output_errors = y - self.h_output_output
        self.h_i_delta_w = learning_rate * X.T.dot(self.h_input_errors)
        self.h_o_delta_w = learning_rate * y.T.dot(self.h_output_errors)
        self.i_hidden_weights += self.h_i_delta_w  # self.h_input_weights
        self.o_hidden_weights += self.h_o_delta_w  # self.h_output_weights

        # from input
        self.i_hidden_activation = np.dot(X, self.i_hidden_weights)
        self.i_hidden_output = normal_CDF(self.i_hidden_activation)

        # from outputs
        self.o_hidden_activation = np.dot(y, self.o_hidden_weights)
        self.o_hidden_output = normal_CDF(self.o_hidden_activation)

        # update i_hidden_weights and o_hidden_weights based on others' output
        self.i_hidden_errors = self.o_hidden_output - self.i_hidden_output
        self.o_hidden_errors = self.i_hidden_output - self.o_hidden_output
        self.i_delta_w = learning_rate * X.T.dot(self.i_hidden_errors)
        self.o_delta_w = learning_rate * y.T.dot(self.o_hidden_errors)
        self.i_hidden_weights += self.i_delta_w
        self.o_hidden_weights += self.o_delta_w
        # biases omitted in this network, but updating them is almost the same:
        #    delta_b = learning_rate * error
        #    self.hidden_bias += delta_b

    def forward(self, X):
        i_hidden_activation = np.dot(X, self.i_hidden_weights)
        i_hidden_output = normal_CDF(i_hidden_activation)
        # self.h_output_weights
        h_output_activation = np.dot(
            i_hidden_output, self.o_hidden_weights)  # .T
        return normal_CDF(h_output_activation)

    def backwards(self, y):
        o_hidden_activation = np.dot(y, self.o_hidden_weights)
        o_hidden_output = normal_CDF(o_hidden_activation)
        # self.h_input_weights
        h_input_activation = np.dot(o_hidden_output, self.i_hidden_weights)
        return normal_CDF(h_input_activation)

    # def suggested_backward(self, X, y, learning_rate=0.1):
    #    # Assume error is the known error attributed to the layer
    #    error = ...  # your error calculation here
    #
    #    # Compute the change in weights and biases
    #    delta_w = learning_rate * error * X.T
    #    delta_b = learning_rate * error
    #
    #    # Update the weights and biases
    #    self.hidden_weights += delta_w
    #    self.hidden_bias += delta_b

    def predict(self, X):
        return np.round(self.forward(X))


# Usage
if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0, 0], [1, 1], [1, 1], [0, 0]])  # XOR problem
    print('learning:\n ', '\n  '.join([f'{i} -> {o}' for i, o in zip(X, y)]))
    nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=2)
    nn.train(X, y)
    nn.train(X, y)
    print('predictions:\n ', '\n  '.join([f'{i} -> {o[0]} ({k[0]})' for i, o,
          k in zip(X, nn.predict(X), nn.forward(X))]))
    print('network:')
    print('  hidden_weights:', '\n    '.join(
        [f'{i}' for i in nn.i_hidden_weights]))
    print('  output_weights:', '\n    '.join(
        [f'{i}' for i in nn.o_hidden_weights]))
