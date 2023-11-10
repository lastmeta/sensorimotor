'''
forward both ways with different weights, -1 to 1, two weights per connection.
(see nest_ff.py)

As I've thought more about this I think we can explore some more simple design
aspects:

input layer | middle1 | middle2 | middle3 | output layer
--------------------------------------------------------

1. edges should weight the incoming data 100 %, while middle2 should weight or 
bias the middle1 the same as middle3. This might mean we allocate some of the 
bias to one or the other layer. idk.
2. I think activations must be bounded back as well as the modifications to the
weights. so that this is the backpropagation process... 

'''
import numpy as np


def tall_normal_CDF(x):
    ''' approximates normal CDF, almost as easy to compute as sigmoid '''
    return 2*(1 / (1 + np.exp(-x/0.55)))-1


def tall_normal_CDF_derivative(x):
    ''' untested '''
    exp_term = np.exp(-x / 0.55)
    return 2 * (exp_term / (0.55 * ((1 + exp_term) ** 2)))


# just use bias instead.
def normal_PDF(x, yStretch: float, xStretch: float):
    ''' standard normal probability density function. untested. '''
    return yStretch * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (xStretch) * x**2)

# just use bias instead.


def normal_PDF_derivative(x):  # doesn't include stretch in derivative yet.
    ''' derivative of the normal probability density function untested '''
    return -x * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)


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

        self.hidden_weights = self.rng.uniform(
            low=0.0, high=1.0, size=(input_size, hidden_size))
        self.hidden_bias = self.rng.uniform(
            low=0.0, high=1.0, size=(1, hidden_size))
        self.output_weights = self.rng.uniform(
            low=0.0, high=1.0, size=(hidden_size, output_size))
        self.output_bias = self.rng.uniform(
            low=0.0, high=1.0, size=(1, output_size))

    def forward(self, X):
        self.hidden_layer_activation = np.dot(
            X, self.hidden_weights) + self.hidden_bias
        self.hidden_layer_output = sigmoid(self.hidden_layer_activation)
        self.output_layer_activation = np.dot(
            self.hidden_layer_output, self.output_weights) + self.output_bias
        self.output = sigmoid(self.output_layer_activation)
        return self.output

    def backward(self, X, y, learning_rate=0.1):
        output_errors = y - self.output
        d_output = output_errors * sigmoid_derivative(self.output)

        hidden_errors = d_output.dot(self.output_weights.T)
        d_hidden_layer = hidden_errors * \
            sigmoid_derivative(self.hidden_layer_output)

        # Updating Weights and Biases
        self.output_weights += self.hidden_layer_output.T.dot(
            d_output) * learning_rate
        self.output_bias += np.sum(d_output, axis=0,
                                   keepdims=True) * learning_rate
        self.hidden_weights += X.T.dot(d_hidden_layer) * learning_rate
        self.hidden_bias += np.sum(d_hidden_layer,
                                   axis=0, keepdims=True) * learning_rate

    def train(self, X, y, learning_rate=0.1, epochs=10000):
        for _ in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)

    def predict(self, X):
        return np.round(self.forward(X))


# Usage
if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR problem
    print('learning:\n ', '\n  '.join([f'{i} -> {o}' for i, o in zip(X, y)]))
    nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)
    nn.train(X, y)
    print('predictions:\n ', '\n  '.join([f'{i} -> {o[0]} ({k[0]})' for i, o,
          k in zip(X, nn.predict(X), nn.forward(X))]))
    print('network:')
    print('  hidden_weights:', '\n    '.join(
        [f'{i}' for i in nn.hidden_weights]))
    print('  hidden_bias:', '\n    '.join(
        [f'{i}' for i in nn.hidden_bias]))
    print('  output_weights:', '\n    '.join(
        [f'{i}' for i in nn.output_weights]))
    print('  output_bias:', '\n    '.join(
        [f'{i}' for i in nn.output_bias]))
