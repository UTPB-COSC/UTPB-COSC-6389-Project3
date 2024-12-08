import numpy as np
import tkinter as tk
from tkinter import Canvas
import gzip
import struct


# Convolutional Layer Implementation
class ConvLayer:
    def __init__(self, num_filters, kernel_size, input_shape):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.kernels = np.random.randn(num_filters, kernel_size, kernel_size) * 0.1
        self.biases = np.zeros((num_filters,))
    
    def forward(self, input_matrix):
        self.input_matrix = input_matrix
        self.output_shape = (
            self.input_shape[0] - self.kernel_size + 1,
            self.input_shape[1] - self.kernel_size + 1,
            self.num_filters,
        )
        self.output_matrix = np.zeros(self.output_shape)

        for f in range(self.num_filters):
            for i in range(self.output_shape[0]):
                for j in range(self.output_shape[1]):
                    region = input_matrix[i:i + self.kernel_size, j:j + self.kernel_size]
                    self.output_matrix[i, j, f] = np.sum(region * self.kernels[f]) + self.biases[f]
        
        return self.output_matrix

    def backward(self, error_matrix, learning_rate):
        kernel_gradients = np.zeros_like(self.kernels)
        bias_gradients = np.zeros_like(self.biases)
        propagated_error = np.zeros_like(self.input_matrix)

        for f in range(self.num_filters):
            for i in range(self.output_shape[0]):
                for j in range(self.output_shape[1]):
                    region = self.input_matrix[i:i + self.kernel_size, j:j + self.kernel_size]
                    kernel_gradients[f] += error_matrix[i, j, f] * region
                    propagated_error[i:i + self.kernel_size, j:j + self.kernel_size] += error_matrix[i, j, f] * self.kernels[f]
                    bias_gradients[f] += error_matrix[i, j, f]
        
        self.kernels -= learning_rate * kernel_gradients
        self.biases -= learning_rate * bias_gradients
        
        return propagated_error


# Pooling Layer Implementation
class PoolingLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, input_matrix):
        self.input_matrix = input_matrix
        self.output_shape = (
            input_matrix.shape[0] // self.pool_size,
            input_matrix.shape[1] // self.pool_size,
            input_matrix.shape[2],
        )
        self.output_matrix = np.zeros(self.output_shape)

        for f in range(input_matrix.shape[2]):
            for i in range(0, input_matrix.shape[0], self.pool_size):
                for j in range(0, input_matrix.shape[1], self.pool_size):
                    region = input_matrix[i:i + self.pool_size, j:j + self.pool_size, f]
                    self.output_matrix[i // self.pool_size, j // self.pool_size, f] = np.max(region)
        
        return self.output_matrix

    def backward(self, error_matrix):
        propagated_error = np.zeros_like(self.input_matrix)

        for f in range(self.output_shape[2]):
            for i in range(0, self.input_matrix.shape[0], self.pool_size):
                for j in range(0, self.input_matrix.shape[1], self.pool_size):
                    region = self.input_matrix[i:i + self.pool_size, j:j + self.pool_size, f]
                    max_val = np.max(region)
                    mask = (region == max_val)
                    propagated_error[i:i + self.pool_size, j:j + self.pool_size, f] += error_matrix[i // self.pool_size, j // self.pool_size, f] * mask
        
        return propagated_error


# Fully Connected Layer Implementation
class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros((output_size,))
    
    def forward(self, input_vector):
        self.input_vector = input_vector
        self.output_vector = np.dot(input_vector, self.weights) + self.biases
        return self.output_vector

    def backward(self, error_vector, learning_rate):
        weights_gradient = np.dot(self.input_vector.T, error_vector)
        propagated_error = np.dot(error_vector, self.weights.T)

        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * np.sum(error_vector, axis=0)
        
        return propagated_error


# Activation Functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


# Categorical Crossentropy Loss
def categorical_crossentropy(predictions, targets):
    return -np.sum(targets * np.log(predictions + 1e-9)) / targets.shape[0]


# Load MNIST Dataset Locally
def load_mnist():
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    def fetch_and_parse(file_name):
        with gzip.open(file_name, 'rb') as file:
            return file.read()

    def parse_images(data):
        _, num, rows, cols = struct.unpack(">IIII", data[:16])
        return np.frombuffer(data[16:], dtype=np.uint8).reshape(num, rows, cols)

    def parse_labels(data):
        _, num = struct.unpack(">II", data[:8])
        return np.frombuffer(data[8:], dtype=np.uint8)

    train_images = parse_images(fetch_and_parse(files["train_images"]))
    train_labels = parse_labels(fetch_and_parse(files["train_labels"]))
    test_images = parse_images(fetch_and_parse(files["test_images"]))
    test_labels = parse_labels(fetch_and_parse(files["test_labels"]))

    return train_images, train_labels, test_images, test_labels


# Visualize Digits Using Tkinter
def visualize_digit(images, labels):
    root = tk.Tk()
    root.title("MNIST Visualization")

    canvas = Canvas(root, width=280, height=280)
    canvas.pack()

    idx = np.random.randint(len(images))
    digit_image = images[idx]
    label = labels[idx]

    for i in range(28):
        for j in range(28):
            grayscale = 255 - digit_image[i, j]
            color = f"#{grayscale:02x}{grayscale:02x}{grayscale:02x}"
            canvas.create_rectangle(j * 10, i * 10, (j + 1) * 10, (i + 1) * 10, fill=color, outline=color)

    label_text = tk.Label(root, text=f"Label: {label}", font=("Arial", 16))
    label_text.pack()

    root.mainloop()


# Main Function
if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = load_mnist()
    visualize_digit(train_images, train_labels)


