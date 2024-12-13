import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split


def load_dataset(dataset_path, image_size=(64, 64)):
    data, labels = [], []
    class_mapping = {"rose": 0, "sun": 1}

    for class_name, label in class_mapping.items():
        class_path = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_path):
            if img_name.endswith((".jpg", ".png")):
                img_path = os.path.join(class_path, img_name)
                img = Image.open(img_path).convert('L')
                img = img.resize(image_size)
                img_array = np.array(img) / 255.0
                data.append(img_array)
                labels.append(label)

    data = np.array(data).reshape(-1, image_size[0], image_size[1], 1)
    labels = np.eye(len(class_mapping))[labels]
    return train_test_split(data, labels, test_size=0.2, random_state=42)


class Convolution:
    """
    A Convolution layer that applies learned filters to an input image.

    This layer implements a basic convolution operation followed by a bias addition.
    No padding or stride > 1 is implemented. Suitable for simple CNN prototypes.
    """

    def __init__(self, input_shape, filter_size, num_filters):

        """
        Initialize the Convolution layer with given parameters and He initialization.

        Args:
            input_shape (tuple): (height, width) of the input image.
            filter_size (int): The height and width of the convolution filters.
            num_filters (int): Number of filters (output channels).
        """

        input_height, input_width = input_shape
        self.num_filters = num_filters
        self.filter_size = filter_size
        if filter_size > input_height or filter_size > input_width:
            raise ValueError("Filter size too large.")
        self.output_shape = (num_filters, input_height - filter_size + 1, input_width - filter_size + 1)
        if self.output_shape[1] <= 0 or self.output_shape[2] <= 0:
            raise ValueError("Invalid output dimensions.")
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * np.sqrt(2 / (filter_size * filter_size))
        self.biases = np.zeros(self.num_filters)
        self.input_data = None

    def forward(self, input_data):
        """
        Perform the forward pass of the convolution operation.

        Args:
            input_data (np.ndarray): Input image of shape (height, width, channels).

        Returns:
            np.ndarray: The output feature map of shape (H_out, W_out, num_filters).
        """
        self.input_data = input_data
        self.output_height = input_data.shape[0] - self.filter_size + 1
        self.output_width = input_data.shape[1] - self.filter_size + 1
        num_channels = input_data.shape[2]

        output = np.zeros((self.output_height, self.output_width, self.num_filters))
        for f in range(self.num_filters):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    input_patch = input_data[i:(i + self.filter_size), j:(j + self.filter_size), :]

                    output[i, j, f] = np.sum(input_patch * self.filters[f]) + self.biases[f]
        return output

    def backward(self, dL_dout, lr):
        """
        Backpropagate through the convolution layer, updating filters and biases.

        Args:
            dL_dout (np.ndarray): Gradient of the loss w.r.t. this layer's output.
            lr (float): Learning rate.

        Returns:
            np.ndarray: Gradient of the loss w.r.t. the input to this layer.
        """
        dL_dinput = np.zeros_like(self.input_data)
        dL_dfilters = np.zeros_like(self.filters)
        for f in range(self.num_filters):
            for i in range(dL_dout.shape[1]):
                for j in range(dL_dout.shape[2]):
                    patch = self.input_data[i:i + self.filter_size, j:j + self.filter_size, 0]
                    dL_dfilters[f] += patch * dL_dout[f, i, j]
                    dL_dinput[i:i + self.filter_size, j:j + self.filter_size, 0] += self.filters[f] * dL_dout[f, i, j]

        # Monitor gradient norms
        filter_grad_norm = np.linalg.norm(dL_dfilters)
        bias_grad_norm = np.linalg.norm(np.sum(dL_dout, axis=(1, 2)))
        print(f"Conv Layer - Filter Gradient Norm: {filter_grad_norm}, Bias Gradient Norm: {bias_grad_norm}")

        # Update weights and biases
        self.filters -= lr * dL_dfilters
        self.biases -= lr * np.sum(dL_dout, axis=(1, 2))
        return dL_dinput


class MaxPool:
    """
    A MaxPooling layer that reduces the spatial dimensions of the input.

    It outputs the maximum value within each pool-size region, helping with spatial invariance.
    """

    def __init__(self, pool_size):
        """
        Initialize the MaxPool layer.

        Args:
            pool_size (int): The size of the pooling window (both height and width).
        """
        self.pool_size = pool_size

    def forward(self, input_data):
        """
        Perform the max-pooling operation on the input data.

        Args:
            input_data (np.ndarray): Input feature map of shape (C, H, W).

        Returns:
            np.ndarray: Reduced feature map of shape (C, H_out, W_out).
        """
        self.input_data = input_data
        self.num_channels, self.input_height, self.input_width = input_data.shape
        self.output_height = self.input_height // self.pool_size
        self.output_width = self.input_width // self.pool_size
        self.output = np.zeros((self.num_channels, self.output_height, self.output_width))
        for c in range(self.num_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size
                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size
                    patch = input_data[c, start_i:end_i, start_j:end_j]
                    self.output[c, i, j] = np.max(patch)
        return self.output

    def backward(self, dL_dout, lr):
        """
        Backpropagate through the MaxPool layer.

        Args:
            dL_dout (np.ndarray): Gradient of the loss w.r.t. this layer's output.
            lr (float): Learning rate (not typically used in pooling).

        Returns:
            np.ndarray: Gradient of the loss w.r.t. the input to this layer.
        """
        dL_dinput = np.zeros_like(self.input_data)
        for c in range(self.num_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size
                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size
                    patch = self.input_data[c, start_i:end_i, start_j:end_j]
                    mask = patch == np.max(patch)
                    dL_dinput[c, start_i:end_i, start_j:end_j] = dL_dout[c, i, j] * mask

        # Optionally monitor the gradients of the pooling layer's input
        input_grad_norm = np.linalg.norm(dL_dinput)
        print(f"Pool Layer - Input Gradient Norm: {input_grad_norm}")

        return dL_dinput


class Fully_Connected:
    """
    A Fully Connected (Dense) layer that transforms the input feature vector into class scores.

    Implements a linear transform followed by a softmax activation for classification.
    """

    def __init__(self, input_size, output_size):
        """
        Initialize the Fully Connected layer with He initialization.

        Args:
            input_size (int): Dimensionality of the input vector.
            output_size (int): Number of classes for the output.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
        self.biases = np.zeros((output_size, 1))

    def softmax(self, z):
        """
        Compute the softmax activation.

        Args:
            z (np.ndarray): Pre-activation logits.

        Returns:
            np.ndarray: Probability distribution over classes.
        """
        shifted_z = z - np.max(z)  # Prevent large exponentials
        exp_values = np.exp(shifted_z)
        sum_exp_values = np.sum(exp_values, axis=0, keepdims=True)
        probabilities = exp_values / sum_exp_values
        return probabilities

    def softmax_derivative(self, s):
        """
        Compute the derivative of the softmax function.

        Args:
            s (np.ndarray): Softmax probabilities.

        Returns:
            np.ndarray: The Jacobian matrix of softmax derivatives.
        """
        return np.diagflat(s) - np.dot(s, s.T)

    def forward(self, input_data):
        """
        Forward pass of the Fully Connected layer.

        Args:
            input_data (np.ndarray): Input feature map (flattened before multiplication).

        Returns:
            np.ndarray: Class probabilities after softmax.
        """
        self.input_data = input_data
        flattened_input = input_data.flatten().reshape(1, -1)
        self.z = np.dot(self.weights, flattened_input.T) + self.biases
        self.output = self.softmax(self.z)
        return self.output

    def backward(self, dL_dout, lr):
        """
        Backward pass through the Fully Connected layer, updating weights and biases.

        Args:
            dL_dout (np.ndarray): Gradient of the loss w.r.t. the output of this layer.
            lr (float): Learning rate.

        Returns:
            np.ndarray: Gradient of the loss w.r.t. the input to this layer.
        """
        dL_dy = np.dot(self.softmax_derivative(self.output), dL_dout)
        dL_dw = np.dot(dL_dy, self.input_data.flatten().reshape(1, -1))
        dL_db = dL_dy
        dL_dinput = np.dot(self.weights.T, dL_dy)
        dL_dinput = dL_dinput.reshape(self.input_data.shape)

        # Monitor gradient norms
        weight_grad_norm = np.linalg.norm(dL_dw)
        bias_grad_norm = np.linalg.norm(dL_db)
        print(f"FC Layer - Weight Gradient Norm: {weight_grad_norm}, Bias Gradient Norm: {bias_grad_norm}")

        # Update weights and biases
        self.weights -= lr * dL_dw
        self.biases -= lr * dL_db
        return dL_dinput

def train_model(X_train, y_train, conv, pool, full, lr, epochs):
    for epoch in range(epochs):
        total_loss, correct_predictions = 0, 0
        for i in range(len(X_train)):
            # Forward pass
            pass

            # Backward pass
            pass
        print(f"Epoch {epoch + 1}/{epochs}: Loss = {total_loss}, Accuracy = {correct_predictions / len(X_train)}")


def cross_entropy_loss(predictions, targets):
    epsilon = 1e-7
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.sum(targets * np.log(predictions)) / targets.shape[0]


def cross_entropy_loss_gradient(actual_labels, predicted_probs):
    epsilon = 1e-7
    predicted_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)
    return -(actual_labels / predicted_probs) / actual_labels.shape[0]
def leaky_relu_derivative(x, alpha=0.01):
    """
    Compute the derivative of the Leaky ReLU activation function.

    Args:
        x (np.ndarray): Input tensor (pre-activation values).
        alpha (float): Negative slope used in Leaky ReLU.

    Returns:
        np.ndarray: The derivative mask, with 1 for x>0 and alpha for x<=0.
    """
    # Derivative is 1 for x > 0, and alpha otherwise
    grad = np.ones_like(x)
    grad[x <= 0] = alpha
    return grad
def leaky_relu(x, alpha=0.01):
    """
    Apply the Leaky ReLU activation function.

    Leaky ReLU sets negative values to `alpha * x` instead of zero, helping to avoid "dead" ReLUs.

    Args:
        x (np.ndarray): Input tensor.
        alpha (float): Negative slope coefficient.

    Returns:
        np.ndarray: Activated output.
    """
    return np.where(x > 0, x, alpha * x)
