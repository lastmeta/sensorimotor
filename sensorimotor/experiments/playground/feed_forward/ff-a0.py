'''
forward both ways with bias, and with different weights and with traditional backprop
'''
import numpy as np


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
