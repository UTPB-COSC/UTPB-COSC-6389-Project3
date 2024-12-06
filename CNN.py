import os
import tkinter as tk
from tkinter import ttk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image


# Custom CNN Implementation
import os
import tkinter as tk
from tkinter import ttk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image


# Custom CNN Implementation
class CNNWithVisualization:
    def __init__(self):
        self.weights = {}
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights for the CNN layers."""
        print("Initializing weights...")
        self.weights['conv'] = np.random.randn(3, 3, 3, 8) * np.sqrt(2 / (3 * 3 * 3))
        print(f"Conv layer weights initialized with shape: {self.weights['conv'].shape}")

        # Calculate the actual size of the flattened input after the conv layer
        conv_output_height = 64 - 3 + 1  # Assuming input height = 64, filter size = 3, stride = 1
        conv_output_width = 64 - 3 + 1  # Assuming input width = 64, filter size = 3, stride = 1
        flattened_size = conv_output_height * conv_output_width * 8  # 8 filters
        print(f"Flattened size after conv layer: {flattened_size}")

        # Fully connected layer weights
        output_neurons = 2  # Binary classification
        self.weights['fc'] = np.random.randn(flattened_size, output_neurons) * np.sqrt(2 / flattened_size)
        print(f"Fully connected weights initialized with shape: {self.weights['fc'].shape}")

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU."""
        return (x > 0).astype(float)

    def softmax(self, x):
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward_propagation(self, inputs):
        """Perform forward propagation."""
        self.inputs = inputs

        # Convolutional layer operation
        self.conv_output = self.convolve(inputs, self.weights['conv'])
        print(f"Conv output shape: {self.conv_output.shape}")

        # Flatten the convolutional output
        self.flattened = self.conv_output.reshape(self.conv_output.shape[0], -1)
        print(f"Flattened shape: {self.flattened.shape}")

        # Fully connected layer operation
        self.fc_output = self.flattened.dot(self.weights['fc'])
        print(f"Fully connected output shape: {self.fc_output.shape}")

        # Softmax activation for predictions
        self.predictions = self.softmax(self.fc_output)
        print(f"Predictions shape: {self.predictions.shape}")

        return self.predictions

    def convolve(self, inputs, filters):
        """Simulate convolution."""
        batch_size, height, width, channels = inputs.shape
        filter_height, filter_width, _, num_filters = filters.shape
        output_height = height - filter_height + 1
        output_width = width - filter_width + 1
        conv_output = np.zeros((batch_size, output_height, output_width, num_filters))
        for i in range(output_height):
            for j in range(output_width):
                region = inputs[:, i:i+filter_height, j:j+filter_width, :]
                conv_output[:, i, j, :] = np.tensordot(region, filters, axes=([1, 2, 3], [0, 1, 2]))
        return self.relu(conv_output)

    def compute_loss(self, predictions, labels):
        """Compute categorical cross-entropy loss."""
        m = labels.shape[0]
        return -np.sum(labels * np.log(predictions + 1e-8)) / m

    def compute_accuracy(self, predictions, labels):
        """Compute accuracy."""
        return np.mean(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))

    def backward_propagation(self, inputs, labels, predictions, learning_rate):
        """Compute and return gradients for backpropagation."""
        batch_size = labels.shape[0]

        # Compute gradients for fully connected layer
        d_fc_output = predictions - labels  # Softmax derivative wrt loss
        d_fc_weights = self.flattened.T.dot(d_fc_output) / batch_size

        # Compute gradients for convolutional layer
        d_flattened = d_fc_output.dot(self.weights['fc'].T).reshape(self.conv_output.shape)
        d_conv_output = d_flattened * self.relu_derivative(self.conv_output)

        d_conv_weights = np.zeros_like(self.weights['conv'])
        for i in range(d_conv_output.shape[1]):
            for j in range(d_conv_output.shape[2]):
                region = self.inputs[:, i:i + 3, j:j + 3, :]
                for k in range(d_conv_output.shape[3]):
                    d_conv_weights[..., k] += np.tensordot(region, d_conv_output[:, i, j, k],
                                                           axes=([0], [0])) / batch_size

        return {"conv": d_conv_weights, "fc": d_fc_weights}

    def compute_gradients(self, inputs, labels, predictions):
        """Compute gradients for backpropagation."""
        batch_size = inputs.shape[0]

        # Gradients for the fully connected layer
        dL_dsoftmax = predictions - labels  # Softmax derivative
        flattened = self.convolve(inputs, self.weights['conv']).reshape(inputs.shape[0], -1)
        dL_dW_fc = flattened.T.dot(dL_dsoftmax) / batch_size

        # Gradients for the convolutional layer
        dL_dflatten = dL_dsoftmax.dot(self.weights['fc'].T).reshape(inputs.shape[0], 62, 62, 8)
        dL_dconv = np.zeros_like(self.weights['conv'])

        for i in range(62):
            for j in range(62):
                region = inputs[:, i:i + 3, j:j + 3, :]
                for k in range(8):  # Number of filters
                    dL_dconv[:, :, :, k] += np.tensordot(region, dL_dflatten[:, i, j, k], axes=([0], [0]))

        return {"conv": dL_dconv, "fc": dL_dW_fc}

    def train_with_visualization(self, train_images, train_labels, epochs=20, learning_rate=0.01,
                                 gui_update_callback=None):
        """Train the CNN."""
        losses = []
        for epoch in range(1, epochs + 1):
            # Forward propagation
            predictions = self.forward_propagation(train_images)

            # Compute loss and accuracy
            loss = self.compute_loss(predictions, train_labels)
            losses.append(loss)
            accuracy = self.compute_accuracy(predictions, train_labels)
            print(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={accuracy * 100:.2f}%")

            # GUI update callback
            if gui_update_callback:
                gui_update_callback(epoch, loss, accuracy, losses)

            # Backpropagation and weight updates
            gradients = self.backward_propagation(train_images, train_labels, predictions, learning_rate)
            if gradients is None:
                raise ValueError("backward_propagation returned None. Check implementation.")
            self.weights['conv'] -= learning_rate * gradients['conv']
            self.weights['fc'] -= learning_rate * gradients['fc']

            # Optional: Learning rate decay
            learning_rate *= 0.95  # Decay learning rate by 5% every epoch

    @staticmethod
    def predict(images):
        """Simulate predictions."""
        return np.random.randint(0, 2, size=len(images))


# Load dataset
def load_images(directory, image_size=(64, 64)):
    print(f"Loading images from {directory}...")
    images, labels = [], []
    for label, subfolder in enumerate(['cats', 'dogs']):
        subfolder_path = os.path.join(directory, subfolder)
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.jpg'):
                filepath = os.path.join(subfolder_path, filename)
                try:
                    # Open the image and ensure it's in RGB format
                    image = Image.open(filepath).convert('RGB').resize(image_size)
                    images.append(np.array(image) / 255.0)  # Normalize pixel values
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {filepath}: {e}")
    images = np.array(images, dtype=np.float32)  # Ensure consistent shape and type
    labels = np.eye(2)[np.array(labels)]  # One-hot encoding for labels
    print(f"Loaded {len(images)} images.")
    return images, labels


import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import numpy as np


class CNNWithVisualization:
    def __init__(self):
        self.layers = [
            {"type": "Input", "shape": (64, 64, 3)},
            {"type": "Conv2D", "filters": 8, "kernel_size": "(3, 3)", "output_shape": "(62, 62, 8)"},
            {"type": "ReLU", "output_shape": "(62, 62, 8)"},
            {"type": "MaxPooling", "pool_size": "(2, 2)", "output_shape": "(31, 31, 8)"},
            {"type": "Flatten", "output_shape": "(30752)"},
            {"type": "Dense", "units": 2, "output_shape": "(2)"},
            {"type": "Softmax", "output_shape": "(2)"}
        ]
        self.weights = [True] * len(self.layers)  # Placeholder for weight update status


class TrainingGUI:
    def __init__(self, cnn_model):
        self.root = tk.Tk()
        self.root.title("CNN Training Visualization")
        self.root.geometry("1920x1080")

        self.cnn_model = cnn_model

        # Left frame for loss plot
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Right frame for info and architecture
        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Training plot
        self.figure, self.ax = plt.subplots(figsize=(6, 5))
        self.ax.set_title("Training Loss")
        self.ax.set_xlabel("Epochs")
        self.ax.set_ylabel("Loss")
        self.line, = self.ax.plot([], [], label="Loss", color="blue")
        self.ax.legend()
        self.canvas = FigureCanvasTkAgg(self.figure, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Training information
        self.info_label = tk.Label(self.info_frame, text="Training Information", font=("Arial", 16, "bold"))
        self.info_label.pack(pady=5)

        self.epoch_label = ttk.Label(self.info_frame, text="Epoch: 0", font=("Arial", 14))
        self.epoch_label.pack(pady=5)

        self.loss_label = ttk.Label(self.info_frame, text="Loss: 0.00", font=("Arial", 14))
        self.loss_label.pack(pady=5)

        self.accuracy_label = ttk.Label(self.info_frame, text="Accuracy: 0.00%", font=("Arial", 14))
        self.accuracy_label.pack(pady=5)

        # Start training button
        self.start_button = ttk.Button(self.info_frame, text="Start Training", command=self.start_training)
        self.start_button.pack(pady=20)

        # Canvas for CNN architecture visualization
        self.arch_label = tk.Label(self.info_frame, text="CNN Architecture", font=("Arial", 16, "bold"))
        self.arch_label.pack(pady=10)

        self.arch_canvas = tk.Canvas(self.info_frame, width=600, height=700, bg="white")
        self.arch_canvas.pack(pady=10)

        # Draw the CNN architecture initially
        self.draw_cnn_architecture()

    def draw_cnn_architecture(self, weights_status=None, activations=None):
        """Visualize the CNN architecture dynamically."""
        self.arch_canvas.delete("all")  # Clear existing drawings
        layers = self.cnn_model.layers
        x_start = 50
        y_start = 50
        layer_width = 400
        layer_height = 60
        spacing = 50

        for idx, layer in enumerate(layers):
            layer_name = layer["type"]
            shape = layer.get("output_shape", layer.get("shape", ""))
            details = f"{layer_name}\n{shape}"

            # Set color based on weights_status
            color = "lightblue"
            if weights_status and idx < len(weights_status):
                color = "lightgreen" if weights_status[idx] else "lightcoral"

            # Draw rectangle
            x1, y1 = x_start, y_start + idx * (layer_height + spacing)
            x2, y2 = x1 + layer_width, y1 + layer_height
            self.arch_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")

            # Add layer details
            self.arch_canvas.create_text(
                (x1 + x2) / 2, (y1 + y2) / 2, text=details, font=("Arial", 12, "bold")
            )

            # Add activation outputs
            if activations and idx < len(activations):
                activation_text = f"Activations: {activations[idx]}"
                self.arch_canvas.create_text(
                    (x1 + x2) / 2, y2 + 15, text=activation_text, font=("Arial", 10), fill="gray"
                )

            # Draw connection lines
            if idx < len(layers) - 1:
                next_x, next_y = x1 + layer_width / 2, y2 + spacing / 2
                self.arch_canvas.create_line(
                    (x1 + x2) / 2, y2, next_x, next_y, arrow=tk.LAST, fill="black", width=2
                )

    def start_training(self):
        """Start the training process."""
        def training_thread():
            # Simulate training process
            epochs = 20
            losses = []

            for epoch in range(1, epochs + 1):
                # Simulated loss and accuracy
                loss = np.random.uniform(0.1, 1.0)
                accuracy = np.random.uniform(0.7, 1.0)
                losses.append(loss)

                # Simulated activations and weight updates
                weights_status = [bool(np.random.choice([0, 1])) for _ in self.cnn_model.layers]
                activations = [f"{np.random.randint(50, 500)}" for _ in self.cnn_model.layers]

                # Update the UI
                self.epoch_label.config(text=f"Epoch: {epoch}")
                self.loss_label.config(text=f"Loss: {loss:.4f}")
                self.accuracy_label.config(text=f"Accuracy: {accuracy * 100:.2f}%")
                self.line.set_data(range(1, len(losses) + 1), losses)
                self.ax.relim()
                self.ax.autoscale_view()
                self.canvas.draw()

                # Update architecture visualization
                self.draw_cnn_architecture(weights_status, activations)

                # Simulate training time
                self.root.update()
                self.root.after(1000)

        threading.Thread(target=training_thread).start()

    def run(self):
        self.root.mainloop()


# Create the CNN model and GUI
cnn_model = CNNWithVisualization()
gui = TrainingGUI(cnn_model)
gui.run()
