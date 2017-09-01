import numpy as np


class NeuralNetwork(object):

    def __init__(self, number_of_inputs, hidden_layer_dimensions, number_of_outputs,
                 epochs=30, learning_rate=0.01, mini_batch_size=30):

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.rate_size_ratio = self.learning_rate / self.mini_batch_size

        self.weights = []
        self.biases = []

        previous_number_of_neurons = number_of_inputs
        for number_of_neurons in hidden_layer_dimensions:
            self.weights.append(np.random.rand(number_of_neurons, previous_number_of_neurons))
            self.biases.append(np.random.rand(number_of_neurons, 1))
            previous_number_of_neurons = number_of_neurons

        self.weights.append(np.random.rand(number_of_outputs, previous_number_of_neurons))
        self.biases.append(np.random.rand(number_of_outputs, 1))

        self.activations = []
        self.activation_derivatives = []
        self.layer_errors = []

    def train(self, x, y):

        for epoch in range(self.epochs):

            for i in range(0, x.shape[0] - 1, self.mini_batch_size):

                i_end = min(i + self.mini_batch_size, x.shape[0] - 1)
                batch_x = x[i:i_end]
                batch_y = y[i:i_end]

                self.train_mini_batch(batch_x, batch_y)

    def predict(self, x):

        self.feed_forward(x, False)
        y = self.activations[-1]

        return np.transpose(y)

    # Feed forward data array x through neural network and store
    # activation calculations for each layer. Note: option to calculate
    # and store is utilized by back propagation process
    def feed_forward(self, x, calculate_derivatives):

        x = np.transpose(x)
        self.activations = [x]
        self.activation_derivatives = []
        if calculate_derivatives:
            self.activation_derivatives.append(sigmoid_prime()(x))

        for i in range(len(self.weights)):

            weight = self.weights[i]
            bias = self.biases[i]

            self.activations.append(feed_forward_layer(self.activations[i], weight, bias))
            if calculate_derivatives:
                self.activation_derivatives.append(sigmoid_prime()(self.activations[-1]))

    def train_mini_batch(self, x, y):

        self.feed_forward(x, True)

        y = np.transpose(y)
        cost_derivative = self.activations[-1] - y
        output_activation_derivative = self.activation_derivatives[-1]

        self.layer_errors = [np.multiply(cost_derivative, output_activation_derivative)]

        for i in range(len(self.weights) - 2, -1, -1):

            self.layer_errors.append(self.calculate_layer_error(i))

        self._adjust_weights_and_biases(x.shape[0])

    def calculate_layer_error(self, layer_index):

        next_layer_weight = self.weights[layer_index + 1]
        next_layer_weight = np.transpose(next_layer_weight)
        next_layer_error = self.layer_errors[self._get_layer_error_index(layer_index + 1)]
        activation_derivative = self.activation_derivatives[layer_index + 1]

        layer_error = np.dot(next_layer_weight, next_layer_error)
        layer_error = np.multiply(layer_error, activation_derivative)

        return layer_error

    def _get_layer_error_index(self, layer_index):
        return len(self.weights) - layer_index - 1

    def _adjust_weights_and_biases(self, mini_batch_size):

        self.rate_size_ratio = self.learning_rate / mini_batch_size

        for i in range(len(self.weights) - 1, -1, -1):

            layer_error = self.layer_errors[self._get_layer_error_index(i)]
            activation = np.transpose(self.activations[i])

            weight_adjustment = self.rate_size_ratio * (np.dot(layer_error, activation))
            self.weights[i] -= weight_adjustment

            bias_adjustment = np.sum(layer_error, axis=1)
            bias_adjustment = self.rate_size_ratio  * bias_adjustment
            self.biases[i] -= np.reshape(bias_adjustment, [bias_adjustment.shape[0], 1])


def feed_forward_layer(input, weight, bias):

    bias_matrix = np.ones([1, input.shape[1]])
    bias_matrix = np.dot(bias, bias_matrix)

    output = np.dot(weight, input) + bias_matrix
    output = sigmoid_function()(output)

    return output


def sigmoid_function():
    func = lambda x: 1 / (1 + np.exp(-x))
    return np.vectorize(func)


def sigmoid_prime():
    func = lambda x: x * (x - 1)
    return np.vectorize(func)
