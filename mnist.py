import tkinter as tk
import numpy as np
import time


# Helper Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)


# Functions to load MNIST binary files
def load_mnist_images(filename):
    with open(filename, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        num_images = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(
            num_images, rows * cols
        )
        return images


def load_mnist_labels(filename):
    with open(filename, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        num_labels = int.from_bytes(f.read(4), "big")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels


# Load MNIST Data (Binary idx format)
x_train = load_mnist_images("train-images-idx3-ubyte")
y_train = load_mnist_labels("train-labels-idx1-ubyte")
x_test = load_mnist_images("t10k-images-idx3-ubyte")
y_test = load_mnist_labels("t10k-labels-idx1-ubyte")

# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode labels
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# Initialize Parameters
input_size = 784  # 28x28 images
hidden_size = 128
output_size = 10

weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
bias_hidden = np.zeros(hidden_size)
weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
bias_output = np.zeros(output_size)


# UI Visualization
class NeuralNetVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Training Visualization")

        # Canvas for Neural Network
        self.canvas_nn = tk.Canvas(root, width=1000, height=600, bg="white")
        self.canvas_nn.grid(row=0, column=0, rowspan=2)

        # Canvas for Training Image
        self.canvas_image = tk.Canvas(root, width=280, height=280, bg="white")
        self.canvas_image.grid(row=0, column=1)

        # Labels for Epoch, Loss, Accuracy
        self.epoch_label = tk.Label(root, text="Epoch: 0", font=("Arial", 14))
        self.epoch_label.grid(row=1, column=1)
        self.loss_label = tk.Label(root, text="Loss: N/A", font=("Arial", 14))
        self.loss_label.grid(row=2, column=1)
        self.accuracy_label = tk.Label(root, text="Accuracy: N/A", font=("Arial", 14))
        self.accuracy_label.grid(row=3, column=0, columnspan=2)

        # Label to show predicted vs actual values
        self.predicted_actual_label = tk.Label(
            root, text="Predicted: N/A | Actual: N/A", font=("Arial", 14)
        )
        self.predicted_actual_label.grid(row=4, column=0, columnspan=2)

        self.layer_label = tk.Label(root, text="", font=("Arial", 14))
        self.layer_label.grid(row=5, column=1)
        self.last_edges = []

    def update_image(self, image, layer_name):
        """Update the training image visualization."""
        self.canvas_image.delete("all")
        for i in range(28):
            for j in range(28):
                color = int(image[i * 28 + j] * 255)
                hex_color = f"#{color:02x}{color:02x}{color:02x}"
                self.canvas_image.create_rectangle(
                    j * 10,
                    i * 10,
                    (j + 1) * 10,
                    (i + 1) * 10,
                    fill=hex_color,
                    outline="",
                )
        self.layer_label.config(text=f"Training Image Layer: {layer_name}")

    def draw_nn(self, input_image, epoch, loss, layer_name, accuracy=None):
        """Draw the neural network based on the current input image."""
        self.canvas_nn.delete("all")
        node_radius = 15
        layers_x = [150, 450, 750]
        layer_sizes = [input_size, hidden_size, output_size]
        layer_names = ["Input Layer", "Hidden Layer", "Output Layer"]

        # Draw layer names
        for layer_idx, x in enumerate(layers_x):
            self.canvas_nn.create_text(
                x,
                20,
                text=layer_names[layer_idx],
                font=("Arial", 16, "bold"),
                fill="black",
            )

        nodes = []
        # Draw nodes for each layer
        for layer_idx, layer_size in enumerate(layer_sizes):
            layer_nodes = []
            for node_idx in range(layer_size):
                x = layers_x[layer_idx]
                y = 80 + node_idx * 40  # might not visualize all nodes well if large
                # Color nodes (input layer: based on image intensity, others just placeholder)
                if layer_idx == 0:  # Input layer
                    # Red intensity based on pixel
                    color = f"#{int(input_image[node_idx]*255):02x}0000"
                else:
                    # Green-based color just to differentiate
                    val = input_image[node_idx] if node_idx < len(input_image) else 0.5
                    color = f"#{int(255 - (val * 255)):02x}00FF"

                layer_nodes.append(
                    self.canvas_nn.create_oval(
                        x - node_radius,
                        y - node_radius,
                        x + node_radius,
                        y + node_radius,
                        fill=color,
                        outline="black",
                    )
                )
            nodes.append(layer_nodes)

        # Draw edges between layers (simple default color)
        self.last_edges.clear()
        for i, layer_nodes_set in enumerate(nodes[:-1]):
            next_layer_nodes = nodes[i + 1]
            for start_node in layer_nodes_set:
                for end_node in next_layer_nodes:
                    edge = self.canvas_nn.create_line(
                        self.canvas_nn.coords(start_node)[2],
                        self.canvas_nn.coords(start_node)[3],
                        self.canvas_nn.coords(end_node)[0],
                        self.canvas_nn.coords(end_node)[1],
                        fill="#D3D3D3",
                        width=2,
                    )
                    self.last_edges.append(edge)

        # Display Epoch, Loss, and Accuracy
        self.epoch_label.config(text=f"Epoch: {epoch}")
        self.loss_label.config(text=f"Loss: {loss:.4f}")
        if accuracy is not None:
            self.accuracy_label.config(text=f"Accuracy: {accuracy:.2f}%")

    def update_accuracy(self, accuracy):
        """Update accuracy in the UI."""
        self.accuracy_label.config(text=f"Accuracy: {accuracy:.2f}%")

    def update_predicted_actual(self, predicted, actual):
        """Update the predicted and actual labels."""
        self.predicted_actual_label.config(
            text=f"Predicted: {predicted} | Actual: {actual}"
        )


# Training Functions
def forward_propagation(x):
    z_hidden = np.dot(x, weights_input_hidden) + bias_hidden
    a_hidden = sigmoid(z_hidden)
    z_output = np.dot(a_hidden, weights_hidden_output) + bias_output
    a_output = softmax(z_output)
    return a_hidden, a_output


def backward_propagation(x, y, a_hidden, a_output, learning_rate):
    global weights_input_hidden, bias_hidden, weights_hidden_output, bias_output
    output_error = a_output - y
    d_weights_hidden_output = np.dot(a_hidden.T, output_error)
    d_bias_output = output_error.sum(axis=0)

    hidden_error = np.dot(output_error, weights_hidden_output.T) * sigmoid_derivative(
        a_hidden
    )
    d_weights_input_hidden = np.dot(x.T, hidden_error)
    d_bias_hidden = hidden_error.sum(axis=0)

    weights_input_hidden -= learning_rate * d_weights_input_hidden
    bias_hidden -= learning_rate * d_bias_hidden
    weights_hidden_output -= learning_rate * d_weights_hidden_output
    bias_output -= learning_rate * d_bias_output


def train_model(epochs, learning_rate, visualizer):
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(len(x_train)):
            x = x_train[i : i + 1]
            y = y_train[i : i + 1]
            a_hidden, a_output = forward_propagation(x)
            epoch_loss += -np.sum(y * np.log(a_output + 1e-9))
            backward_propagation(x, y, a_hidden, a_output, learning_rate)

            # Update visualization every 100 steps
            if i % 100 == 0:
                predicted = np.argmax(a_output)
                actual = np.argmax(y)

                current_layer = "Hidden Layer" if i % 200 == 0 else "Output Layer"
                visualizer.update_image(
                    x[0], layer_name="Input Layer" if i == 0 else current_layer
                )
                visualizer.update_predicted_actual(predicted, actual)
                visualizer.draw_nn(
                    x[0], epoch + 1, epoch_loss / (i + 1), layer_name=current_layer
                )
                visualizer.root.update()
                time.sleep(0.05)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Calculate accuracy
    correct_predictions = 0
    for i in range(len(x_test)):
        a_hidden, a_output = forward_propagation(x_test[i : i + 1])
        if np.argmax(a_output) == np.argmax(y_test[i]):
            correct_predictions += 1
    accuracy = (correct_predictions / len(x_test)) * 100
    visualizer.update_accuracy(accuracy)


# Main Code
root = tk.Tk()
visualizer = NeuralNetVisualizer(root)
train_model(epochs=5, learning_rate=0.01, visualizer=visualizer)
root.mainloop()
