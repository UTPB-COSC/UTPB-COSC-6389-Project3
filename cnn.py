import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def softmax_derivative(output, target):
    return output - target


class FCLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros(output_size)

    def forward(self, input_data):
        self.input_data = input_data
        self.output = np.dot(input_data, self.weights) + self.biases
        return self.output

    def backward(self, grad_output):
        self.grad_weights = np.dot(self.input_data.T, grad_output)
        self.grad_biases = grad_output.sum(axis=0)
        grad_input = np.dot(grad_output, self.weights.T)
        return grad_input

    def update(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases


class SimpleCNN:
    def __init__(self, input_shape, num_classes):
        self.fc1 = FCLayer(input_size=np.prod(input_shape), output_size=10)
        self.fc2 = FCLayer(input_size=10, output_size=num_classes)

    def forward(self, input_data):
        out = input_data.reshape(input_data.shape[0], -1)
        out = self.fc1.forward(out)
        out = sigmoid(out)
        return self.fc2.forward(out)

    def backward(self, grad_output):
        grad = self.fc2.backward(grad_output)
        grad = sigmoid_derivative(grad)
        return self.fc1.backward(grad)

    def update(self, learning_rate):
        self.fc1.update(learning_rate)
        self.fc2.update(learning_rate)
