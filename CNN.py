import numpy as np
from scipy.signal import correlate2d
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle
import tkinter as tk
from tkinter import ttk
from matplotlib import colors
from matplotlib import rcParams

#------------------------------------------------------------------------------------------#

def load_dataset(dataset_path, image_size=(28, 28)):
    """Load and preprocess the dataset of images for the classification task.

    Args:
        dataset_path (str): The path to the dataset directory.
        image_size (tuple): The desired (width, height) of the images.

    Returns:
        (X_train, X_test, y_train, y_test): Numpy arrays of training and testing samples and labels.
    """
    data = []
    labels = []
    class_mapping = {
        "glasses": {
            "label": 0,
            "ranges": [(1, 4, 1, 12), (5, 5, 1, 4)]
        },
        "noglasses": {
            "label": 1,
            "ranges": [(7, 10, 1, 12), (11, 11, 1, 4)]
        }
    }

    for class_name, class_info in class_mapping.items():
        label = class_info["label"]
        ranges = class_info["ranges"]
        class_path = os.path.join(dataset_path, class_name)

        for img_name in os.listdir(class_path):
            try:
                parts = img_name.split('-')
                row_number = int(parts[1])
                col_number = int(parts[3].split('.')[0])

                valid_file = False
                for row_start, row_end, col_start, col_end in ranges:
                    if row_start <= row_number <= row_end and col_start <= col_number <= col_end:
                        valid_file = True
                        break

                if valid_file and img_name.endswith(".jpg"):
                    img_path = os.path.join(class_path, img_name)
                    img = Image.open(img_path).convert('L')  
                    img = img.resize(image_size)  
                    img_array = np.array(img) / 255.0  
                    data.append(img_array)
                    labels.append(label)
            except Exception as e:
                print(f"Error processing file {img_name}: {e}")

    data = np.array(data).reshape(-1, image_size[0], image_size[1], 1)
    labels = np.array(labels)

    num_classes = len(class_mapping)
    labels_one_hot = np.zeros((labels.size, num_classes))
    labels_one_hot[np.arange(labels.size), labels] = 1

    return train_test_split(data, labels_one_hot, test_size=0.2, random_state=42)

def clip_gradients(grad, clip_value):
    """Clip the gradients to a specified range to prevent exploding gradients.

    Args:
        grad (np.ndarray): The gradient array to be clipped.
        clip_value (float): The maximum absolute value for gradients.

    Returns:
        np.ndarray: The clipped gradient array.
    """
    return np.clip(grad, -clip_value, clip_value)

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
                    patch = self.input_data[i:i+self.filter_size, j:j+self.filter_size, 0]
                    dL_dfilters[f] += patch * dL_dout[f, i, j]
                    dL_dinput[i:i+self.filter_size, j:j+self.filter_size, 0] += self.filters[f] * dL_dout[f, i, j]
        
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

def cross_entropy_loss(predictions, targets):
    """
    Compute the cross-entropy loss for classification.

    Args:
        predictions (np.ndarray): Predicted probabilities.
        targets (np.ndarray): One-hot encoded true labels.

    Returns:
        float: The average cross-entropy loss.
    """
    epsilon = 1e-7  # Prevent log(0)
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    loss = -np.sum(targets * np.log(predictions)) / targets.shape[0]
    return loss

def cross_entropy_loss_gradient(actual_labels, predicted_probs):
    """
    Compute the gradient of cross-entropy loss w.r.t. predictions.

    Args:
        actual_labels (np.ndarray): One-hot encoded true labels.
        predicted_probs (np.ndarray): Predicted probabilities.

    Returns:
        np.ndarray: Gradient of the loss w.r.t. predictions.
    """
    epsilon = 1e-7
    predicted_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)
    return -(actual_labels / predicted_probs) / actual_labels.shape[0]

class TrainerApp:
    """
    A Trainer application class for visualizing CNN training in real time with a Tkinter GUI.

    Displays:
    - Network structure diagram
    - Real-time loss and accuracy plots
    - Weight updates visualization
    - Status messages and training progress
    """

    def __init__(self, master, X_train, y_train, conv, pool, full, lr=0.01, epochs=10):
        """
        Initialize the TrainerApp with data, model layers, and training parameters.

        Args:
            master (tk.Tk): The Tkinter root window.
            X_train (np.ndarray): Training images.
            y_train (np.ndarray): One-hot encoded training labels.
            conv (Convolution): The convolution layer.
            pool (MaxPool): The pooling layer.
            full (Fully_Connected): The fully connected output layer.
            lr (float): Learning rate.
            epochs (int): Number of epochs to train.
        """
        rcParams.update({
            "axes.facecolor": "black",
            "axes.edgecolor": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "figure.facecolor": "black",
            "figure.edgecolor": "black",
            "grid.color": "gray",
        })

        self.master = master
        self.X_train = X_train
        self.y_train = y_train
        self.conv = conv
        self.pool = pool
        self.full = full
        self.lr = lr
        self.epochs = epochs
        self.current_epoch = 0
        
        self.pool_nodes = conv.num_filters  # Pool nodes match the number of filters
        self.fc_nodes = full.output_size   # Fully connected layer's output size

        # Initialize loss and accuracy tracking
        self.losses = []
        self.accuracies = []

        self.master.title(" 😎CNN Visualization🙂 ")

        # Frames
        main_frame = tk.Frame(self.master, bg="black")
        main_frame.pack(fill=tk.BOTH, expand=True)

        top_frame = tk.Frame(main_frame, bg="black")
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        bottom_frame = tk.Frame(main_frame, bg="black")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        left_plot_frame = tk.Frame(top_frame, bg="black")
        right_plot_frame = tk.Frame(top_frame, bg="black")
        left_plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Status Label
        self.status_label = tk.Label(
            bottom_frame, 
            text="No training started yet.", 
            font=("Helvetica", 20, "bold"),
            bg="black", 
            fg="white"
        )
        self.status_label.pack(pady=30, side=tk.LEFT)

        # Network Figure
        self.fig_network, self.ax_network = plt.subplots(figsize=(4, 4))
        self.ax_network.set_title("CNN Network")
        self.ax_network.axis('off')
        # Store node positions for each layer
        input_positions = self.draw_layer(self.ax_network, 1, 5, 'red', spacing=2.75, y_offset=8)
        conv_positions = self.draw_layer(self.ax_network, 2, self.conv.num_filters, 'blue', spacing=2.5, y_offset=-2)
        pool_positions = self.draw_layer(self.ax_network, 3, self.conv.num_filters, 'green', spacing=2.25, y_offset=6)
        fc_positions = self.draw_layer(self.ax_network, 4, self.full.output_size, 'orange', spacing=2, y_offset=-2)
        # Connect the layers and store connections
        self.connect_layers(self.ax_network, input_positions, conv_positions)
        self.connect_layers(self.ax_network, conv_positions, pool_positions)
        self.connection_lines = self.connect_layers(self.ax_network, pool_positions, fc_positions) 
        self.network_canvas = FigureCanvasTkAgg(self.fig_network, master=left_plot_frame)
        self.network_canvas.draw()
        self.network_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Training Plots
        self.fig, (self.ax_loss, self.ax_accuracy) = plt.subplots(2, 1, figsize=(4, 4))
        self.fig.tight_layout()
        self.loss_line, = self.ax_loss.plot([], [], color="cyan", label="Loss", linewidth=2, linestyle="--")
        self.accuracy_line, = self.ax_accuracy.plot([], [], color="lime", label="Accuracy", linewidth=2, linestyle="-.")
        self.ax_loss.grid(True, linestyle="--", alpha=0.5)
        self.ax_accuracy.grid(True, linestyle="--", alpha=0.5)
        self.ax_loss.set_title("Training Loss", fontsize=20, fontweight="bold", color="white")
        self.ax_accuracy.set_title("Training Accuracy", fontsize=20, fontweight="bold", color="white")
        self.ax_loss.legend(loc="upper right", fontsize=20, facecolor="black", edgecolor="white")
        self.ax_accuracy.legend(loc="lower right", fontsize=20, facecolor="black", edgecolor="white")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Start Button
        self.start_button = ttk.Button(
            bottom_frame, 
            text="Start Training", 
            command=self.start_training, 
            style="TButton"
        )
        self.start_button.pack(pady=25, padx=30, side=tk.RIGHT)

        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 24, "bold"), padding=10)

    def draw_layer(self, ax, x, num_nodes, color='blue', y_offset=0, spacing=0.7):
        """
        Draw a vertical column of circular nodes representing a layer.

        Args:
            ax (matplotlib.axes.Axes): The axes on which to draw.
            x (float): The x-position of the layer.
            num_nodes (int): Number of nodes in this layer.
            color (str): Color of the nodes.
            y_offset (float): Vertical offset to position nodes.
            spacing (float): Vertical spacing between nodes.

        Returns:
            list of tuple: Positions of the nodes.
        """
        positions = []
        for i in range(num_nodes):
            y = i * spacing + y_offset
            circle = Circle((x, y), 0.1, color=color, fill=True)
            ax.add_patch(circle)
            positions.append((x,y))
        return positions

    def connect_layers(self, ax, from_positions, to_positions):
        """
        Draw lines representing connections between two layers.

        Args:
            ax (matplotlib.axes.Axes): The axes on which to draw.
            from_positions (list): Positions of nodes in the preceding layer.
            to_positions (list): Positions of nodes in the next layer.

        Returns:
            list: A list of line objects representing the connections.
        """
        lines = []
        for (x1, y1) in from_positions:
            for (x2, y2) in to_positions:
                line, = ax.plot([x1, x2], [y1, y2], color='gray', linewidth=0.5)
                lines.append(line)
        return lines

    def interpolate_color(self, weight, min_weight, max_weight):
        """
        Interpolate a color between purple and cyan based on the weight value.

        Args:
            weight (float): The weight value.
            min_weight (float): Minimum weight in all connections.
            max_weight (float): Maximum weight in all connections.

        Returns:
            str: A hex color code representing the interpolated color.
        """
        # Interpolate between purple (#FF00FF) and cyan (#00FFFF)
        silver = colors.hex2color("#FF00FF")  # purple
        fire = colors.hex2color("#00FFFF")    # cyan
        if max_weight == min_weight:
            normalized_weight = 0.5
        else:
            normalized_weight = (weight - min_weight) / (max_weight - min_weight)
        interpolated_color = [
            silver[i] + (fire[i] - silver[i]) * normalized_weight
            for i in range(3)
        ]
        return colors.rgb2hex(interpolated_color)

    def update_connections(self):
        """
        Update the color and thickness of connection lines based on the current weights.
        """
        weights = self.full.weights  # shape: (output_size, input_size)
        min_weight = np.min(weights)
        max_weight = np.max(weights)

        line_index = 0
        for i in range(self.pool_nodes):
            for j in range(self.fc_nodes):
                w = weights[j, i]
                color = self.interpolate_color(w, min_weight, max_weight)
                weight_width = min(max(abs(w) * 5, 0.5), 5)  # scale line width with weight magnitude
                line = self.connection_lines[line_index]
                line.set_color(color)
                line.set_linewidth(weight_width)
                line_index += 1

        self.network_canvas.draw()

    def start_training(self):
        """
        Initialize the training process and schedule the first epoch.
        
        """
        self.start_button.config(state="disabled")
        self.status_label.config(text="👓Training started...")
        self.master.after(100, self.run_training_step)

    def run_training_step(self):
        """
        Execute one epoch of training, update the GUI, and schedule the next epoch.
        
        """
        if self.current_epoch < self.epochs:
            total_loss = 0.0
            correct_predictions = 0

            for i in range(len(self.X_train)):
                # Forward pass:
                conv_out = self.conv.forward(self.X_train[i])
                conv_out = np.transpose(conv_out, (2, 0, 1))  # (C, H, W)

                print("Before ReLU:", conv_out)  # Debugging pre-ReLU values
                relu_mask = leaky_relu_derivative(conv_out, alpha=0.01)
                conv_out = leaky_relu(conv_out, alpha=0.01)
                print("After ReLU:", conv_out)  # Debugging post-ReLU values

                pool_out = self.pool.forward(conv_out)
                full_out = self.full.forward(pool_out)

                # Compute loss and accuracy:
                loss = cross_entropy_loss(full_out.flatten(), self.y_train[i])
                total_loss += loss
                predicted_label = np.argmax(full_out)
                true_label = np.argmax(self.y_train[i])
                if predicted_label == true_label:
                    correct_predictions += 1

                # Backward pass:
                gradient = cross_entropy_loss_gradient(self.y_train[i], full_out.flatten()).reshape((-1, 1))
                full_back = self.full.backward(gradient, self.lr)
                pool_back = self.pool.backward(full_back, self.lr)

                # Multiply by the leaky ReLU derivative (stored in relu_mask)
                pool_back = pool_back * relu_mask

                conv_back = self.conv.backward(pool_back, self.lr)




            avg_loss = total_loss / len(self.X_train)
            accuracy = correct_predictions / len(self.X_train) * 100
            self.losses.append(avg_loss)
            self.accuracies.append(accuracy)

            # Update the training plots
            self.loss_line.set_data(range(len(self.losses)), self.losses)
            self.accuracy_line.set_data(range(len(self.accuracies)), self.accuracies)
            self.ax_loss.relim()
            self.ax_loss.autoscale_view()
            self.ax_accuracy.relim()
            self.ax_accuracy.autoscale_view()
            self.canvas.draw()

            # Update connection colors based on current weights
            self.update_connections()

            # Update status label
            self.status_label.config(text=f"Epoch {self.current_epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            print(f"Epoch {self.current_epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

            self.current_epoch += 1
            self.master.after(100, self.run_training_step)
        else:
            final_loss = self.losses[-1]
            final_accuracy = self.accuracies[-1]
            print(f"Training complete! Final Epoch - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.2f}%")
            self.status_label.config(text=f"Training complete! Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.2f}%")
            self.status_label.config(text=" Training completed!  😎🙂")
            self.start_button.config(state="normal")

if __name__ == "__main__":

    try:
        current_dir = os.path.dirname(__file__)
        dataset_path = os.path.join(current_dir, "train")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError("Dataset not found.")

        X_train, X_test, y_train, y_test = load_dataset(dataset_path)
        input_height, input_width = X_train[0].shape[:2]

        conv = Convolution((input_height, input_width), filter_size=3, num_filters=6)
        pool = MaxPool(pool_size=2)
        pool_output_height = conv.output_shape[1] // pool.pool_size
        pool_output_width = conv.output_shape[2] // pool.pool_size
        fully_connected_input_size = conv.num_filters * pool_output_height * pool_output_width

        full = Fully_Connected(input_size=fully_connected_input_size, output_size=2)

        root = tk.Tk()
        root.title("CNN Visualization")
        root.state('zoomed')
        app = TrainerApp(root, X_train, y_train, conv, pool, full, lr=0.01, epochs=10)
        root.mainloop()

    except Exception as e:
        print(f"An error occurred: {e}")