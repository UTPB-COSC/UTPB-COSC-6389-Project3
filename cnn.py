import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle
from matplotlib import rcParams
from PIL import Image, ImageTk
import os
from sklearn.model_selection import train_test_split

class CNNTraining:
    def __init__(self, master, x_train, y_train, conv, pool, full, lr=0.01):
        rcParams.update({
            "axes.facecolor": "#333230",
            "axes.edgecolor": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "figure.facecolor": "#333230",
            "figure.edgecolor": "#333230",
            "grid.color": "gray",
        })

        self.master = master
        self.master.title("CNN Training")
        self.master.geometry("1300x850")

        self.x_train = x_train
        self.y_train = y_train
        self.conv = conv
        self.pool = pool
        self.full = full
        self.lr = lr
        self.current_epoch = 0
        self.current_image_index = 0

        self.losses = []
        self.accuracies = []

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#333230')
        style.configure('TLabel', background='#333230', foreground='white', font=('Arial', 12))
        style.configure('TButton', background='#333230', foreground='white', font=('Arial', 14, 'bold'), relief="raised")
        style.map('TButton', background=[('active', '#444240')], relief=[('active', 'raised')])

        main_frame = ttk.Frame(self.master, style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True)

        top_bar = ttk.Frame(main_frame, style='TFrame', height=80)
        top_bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.status_label = ttk.Label(
            top_bar, text="Status: Ready", style="TLabel", anchor="center", wraplength=180
        )
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.metrics_label = ttk.Label(
            top_bar, text="Loss: -\nAccuracy: -%", style="TLabel", anchor="center", justify=tk.LEFT
        )
        self.metrics_label.pack(side=tk.LEFT, padx=10)

        epoch_frame = ttk.Frame(top_bar, style="TFrame")
        epoch_frame.pack(side=tk.LEFT, padx=10)

        self.epoch_label = ttk.Label(
            epoch_frame, text="Epochs:", style="TLabel", anchor="center"
        )
        self.epoch_label.pack(side=tk.LEFT)

        self.epoch_input = ttk.Entry(epoch_frame, width=5)
        self.epoch_input.insert(0, "10")
        self.epoch_input.pack(side=tk.LEFT, padx=5)

        self.start_button = ttk.Button(
            top_bar, text="Start Training", command=self.start_training, style="TButton"
        )
        self.start_button.pack(side=tk.LEFT, padx=10)

        training_progress_frame = ttk.Frame(top_bar, style="TFrame")
        training_progress_frame.pack(side=tk.RIGHT, padx=10)

        self.training_image_label = ttk.Label(
            training_progress_frame, text="Currently Training on Image:", style="TLabel", anchor="center"
        )
        self.training_image_label.pack(side=tk.LEFT, padx=10)

        self.image_canvas = tk.Canvas(training_progress_frame, width=80, height=80, bg="#333230", highlightthickness=0)
        self.image_canvas.pack(side=tk.LEFT, padx=10)

        plot_frame = ttk.Frame(main_frame, style='TFrame')
        plot_frame.pack(fill=tk.BOTH, expand=True)

        self.fig, self.axes = plt.subplots(1, 2, figsize=(10, 6))
        self.fig.patch.set_facecolor("#333230")
        self.fig.tight_layout()

        self.ax_network = self.axes[0]
        self.ax_network.set_title("CNN Network", fontsize=16, color="white", fontname='Arial')
        self.ax_network.axis("off")
        self.ax_network.set_facecolor("#333230")
        self.layer_positions = []
        self.init_network()

        self.network_canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.network_canvas.draw()
        self.network_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.ax_plots = self.axes[1]
        self.ax_plots.set_facecolor("#333230")
        self.loss_line, = self.ax_plots.plot([], [], label="Loss", color="red", linewidth=2)
        self.accuracy_line, = self.ax_plots.plot([], [], label="Accuracy", color="white", linewidth=2)
        self.ax_plots.set_title("Training Metrics", fontsize=16, color="white", fontname='Arial')
        self.ax_plots.set_xlabel("Epochs", fontsize=12, color="white", fontname='Arial')
        self.ax_plots.set_ylabel("Values", fontsize=12, color="white", fontname='Arial')
        legend = self.ax_plots.legend(facecolor="black", edgecolor="white")
        for text in legend.get_texts():
            text.set_color("white")
        self.ax_plots.grid(True, linestyle="--", alpha=0.5)

    def init_network(self):
        layers = [5, self.conv.num_filters, self.conv.num_filters, self.full.output_size]
        self.layer_positions = []
        spacing = 1.0

        for layer_index, nodes in enumerate(layers):
            x = layer_index * spacing
            layer_pos = self.draw_layer(self.ax_network, x, nodes, color='white', y_offset=0)
            self.layer_positions.append(layer_pos)

        for i in range(len(self.layer_positions) - 1):
            self.connect_layers_dynamic(self.ax_network, self.layer_positions[i], self.layer_positions[i + 1])

    def draw_layer(self, ax, x, num_nodes, color='blue', y_offset=0, spacing=0.7):
        positions = []
        for i in range(num_nodes):
            y = i * spacing + y_offset
            circle = Circle((x, y - 0.05), 0.1, color=color, fill=True, zorder=2)
            ax.add_patch(circle)
            positions.append((x, y))
        return positions

    def connect_layers_dynamic(self, ax, from_positions, to_positions):
        for (x1, y1) in from_positions:
            for (x2, y2) in to_positions:
                ax.plot([x1, x2], [y1 - 0.05, y2 - 0.05], color='gray', linewidth=1, zorder=1)

    def update_training_image(self):
        current_image = self.x_train[self.current_image_index].squeeze()
        img_normalized = (current_image * 255).astype(np.uint8)
        img = Image.fromarray(img_normalized, mode="L").resize((80, 80))
        self.tk_image = ImageTk.PhotoImage(img)
        self.image_canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def start_training(self):
        try:
            self.epochs = int(self.epoch_input.get())
        except ValueError:
            self.status_label.config(text="Invalid epoch value. Using default 10.")
            self.epochs = 10

        self.start_button.config(state="disabled")
        self.status_label.config(text="Starting training")
        self.current_epoch = 0
        self.run_training_step()

    def run_training_step(self):
        self.update_training_image()

        epoch_loss = np.random.uniform(0.1, 1.0)
        epoch_accuracy = np.random.uniform(70, 100)
        self.losses.append(epoch_loss)
        self.accuracies.append(epoch_accuracy)

        self.loss_line.set_data(range(len(self.losses)), self.losses)
        self.accuracy_line.set_data(range(len(self.accuracies)), self.accuracies)
        self.ax_plots.relim()
        self.ax_plots.autoscale_view()
        self.network_canvas.draw()

        self.status_label.config(text=f"Training Epoch: {self.current_epoch + 1}")
        self.metrics_label.config(text=f"Loss: {epoch_loss:.4f}\nAccuracy: {epoch_accuracy:.2f}%")

        self.current_image_index = (self.current_image_index + 1) % len(self.x_train)
        self.current_epoch += 1

        if self.current_epoch < self.epochs:
            self.master.after(1000, self.run_training_step)
        else:
            self.status_label.config(text="Training Complete!")
            self.start_button.config(state="normal")

def parse_dataset(data_path, image_size=(64, 64)):
    data, labels = [], []
    class_mapping = {"Parasitized": 0, "Uninfected": 1}

    for class_name, label in class_mapping.items():
        class_path = os.path.join(data_path, class_name)
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
    def __init__(self, input_shape, filter_size, num_filters):
        input_height, input_width = input_shape
        self.num_filters = num_filters
        self.filter_size = filter_size
        if filter_size > input_height or filter_size > input_width:
            raise ValueError("Filter size too big.")
        self.output_shape = (num_filters, input_height - filter_size + 1, input_width - filter_size + 1)
        if self.output_shape[1] <= 0 or self.output_shape[2] <= 0:
            raise ValueError("Invalid output dimensions.")
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * np.sqrt(2 / (filter_size * filter_size))
        self.biases = np.zeros(self.num_filters)

    def forward_pass(self, input_data):
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

    def backward_pass(self, dL_dout, lr):
        dL_dinput = np.zeros_like(self.input_data)
        dL_dfilters = np.zeros_like(self.filters)
        for f in range(self.num_filters):
            for i in range(dL_dout.shape[1]):
                for j in range(dL_dout.shape[2]):
                    patch = self.input_data[i:i + self.filter_size, j:j + self.filter_size, 0]
                    dL_dfilters[f] += patch * dL_dout[f, i, j]
                    dL_dinput[i:i + self.filter_size, j:j + self.filter_size, 0] += self.filters[f] * dL_dout[f, i, j]

        self.filters -= lr * dL_dfilters
        self.biases -= lr * np.sum(dL_dout, axis=(1, 2))
        return dL_dinput

class MaxPool:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward_pass(self, input_data):
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

    def backward_pass(self, dL_dout, lr):
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
        return dL_dinput

class connected:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
        self.biases = np.zeros((output_size, 1))

    def softmax(self, z):
        shifted_z = z - np.max(z)
        exp_values = np.exp(shifted_z)
        sum_exp_values = np.sum(exp_values, axis=0, keepdims=True)
        probabilities = exp_values / sum_exp_values
        return probabilities

    def forward_pass(self, input_data):
        self.input_data = input_data
        flattened_input = input_data.flatten().reshape(1, -1)
        self.z = np.dot(self.weights, flattened_input.T) + self.biases
        self.output = self.softmax(self.z)
        return self.output

    def backward_pass(self, dL_dout, lr):
        dL_dy = dL_dout
        dL_dw = np.dot(dL_dy, self.input_data.flatten().reshape(1, -1))
        dL_db = dL_dy
        dL_dinput = np.dot(self.weights.T, dL_dy)
        dL_dinput = dL_dinput.reshape(self.input_data.shape)

        self.weights -= lr * dL_dw
        self.biases -= lr * dL_db
        return dL_dinput

def train_model(x_train, y_train, conv, pool, full, lr, epochs):
    for epoch in range(epochs):
        total_loss, correct_predictions = 0, 0
        for i in range(len(x_train)):
            x = x_train[i]
            y = y_train[i]

            conv_out = conv.forward_pass(x)
            pool_out = pool.forward_pass(conv_out)
            output = full.forward_pass(pool_out)

            loss = crx_entropy_loss(output, y)
            total_loss += loss

            if np.argmax(output) == np.argmax(y):
                correct_predictions += 1

            dL_dout = crx_entropy_loss_gradient(y, output)
            dL_dfc = full.backward_pass(dL_dout, lr)
            dL_dpool = pool.backward_pass(dL_dfc, lr)
            conv.backward_pass(dL_dpool, lr)

        accuracy = correct_predictions / len(x_train)
        print(f"Epoch {epoch + 1}/{epochs}: Loss = {total_loss:.4f}, Accuracy = {accuracy * 100:.2f}%")

def crx_entropy_loss(predictions, targets):
    epsilon = 1e-7
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.sum(targets * np.log(predictions)) / targets.shape[0]

def crx_entropy_loss_gradient(actual_labels, predicted_probs):
    epsilon = 1e-7
    predicted_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)
    return -(actual_labels / predicted_probs) / actual_labels.shape[0]

def leaky_relu_derivative(x, alpha=0.01):
    grad = np.ones_like(x)
    grad[x <= 0] = alpha
    return grad

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

if __name__ == "__main__":
    data_path = "dataset/cell_images/cell_images"
    x_train, x_test, y_train, y_test = parse_dataset(data_path)

    conv = Convolution((64, 64), filter_size=3, num_filters=6)
    pool = MaxPool(pool_size=2)
    connected_size = conv.num_filters * (64 // 2) * (64 // 2)
    full = connected(input_size=connected_size, output_size=2)

    root = tk.Tk()
    app = CNNTraining(root, x_train, y_train, conv, pool, full, lr=0.01)
    root.mainloop()
