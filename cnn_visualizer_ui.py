import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle
from matplotlib import rcParams


class TrainerApp:
    """
    Trainer application with enhanced UI for visualizing CNN training in real-time.
    """

    def __init__(self, master, X_train, y_train, conv, pool, full, lr=0.01, epochs=10):
        """
        Initialize the TrainerApp with data, model layers, and training parameters.
        """
        rcParams.update({
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            "grid.color": "gray",
        })

        self.master = master
        self.master.title("CNN Training Visualizer")
        self.master.geometry("1200x800")

        self.X_train = X_train
        self.y_train = y_train
        self.conv = conv
        self.pool = pool
        self.full = full
        self.lr = lr
        self.epochs = epochs
        self.current_epoch = 0

        # Initialize loss and accuracy tracking
        self.losses = []
        self.accuracies = []

        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='white', font=('Helvetica', 12))
        style.configure('TButton', background='#1c1c1c', foreground='white', font=('Helvetica', 14, 'bold'))

        # Main layout
        main_frame = ttk.Frame(self.master, style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Sidebar for controls
        sidebar = ttk.Frame(main_frame, style='TFrame', width=200)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Status Label
        self.status_label = ttk.Label(
            sidebar, text="Status: Ready", style="TLabel", anchor="center", wraplength=180
        )
        self.status_label.pack(pady=20)

        # Start Button
        self.start_button = ttk.Button(
            sidebar, text="Start Training", command=self.start_training, style="TButton"
        )
        self.start_button.pack(pady=20)

        # Metrics
        self.metrics_label = ttk.Label(
            sidebar, text="Loss: -\nAccuracy: -%", style="TLabel", anchor="center", justify=tk.LEFT
        )
        self.metrics_label.pack(pady=20)

        # Plotting area
        plot_frame = ttk.Frame(main_frame, style='TFrame')
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Grid layout for plots
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.tight_layout()

        # Network Diagram
        self.ax_network = self.axes[0]
        self.ax_network.set_title("CNN Network", fontsize=16, color="white")
        self.ax_network.axis("off")

        # Draw Layers
        self.input_positions = self.draw_layer(self.ax_network, x=1, num_nodes=5, color='skyblue', y_offset=0)
        self.conv_positions = self.draw_layer(self.ax_network, x=2, num_nodes=self.conv.num_filters, color='royalblue', y_offset=1)
        self.pool_positions = self.draw_layer(self.ax_network, x=3, num_nodes=self.conv.num_filters, color='limegreen', y_offset=0)
        self.fc_positions = self.draw_layer(self.ax_network, x=4, num_nodes=self.full.output_size, color='gold', y_offset=1)
        self.connect_layers(self.ax_network, self.input_positions, self.conv_positions)
        self.connect_layers(self.ax_network, self.conv_positions, self.pool_positions)
        self.connect_layers(self.ax_network, self.pool_positions, self.fc_positions)

        self.network_canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.network_canvas.draw()
        self.network_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Training Plots (Loss and Accuracy)
        self.ax_plots = self.axes[1]
        self.loss_line, = self.ax_plots.plot([], [], label="Loss", color="cyan", linestyle="--")
        self.accuracy_line, = self.ax_plots.plot([], [], label="Accuracy", color="lime", linestyle="-.")
        self.ax_plots.set_title("Training Metrics", fontsize=16, color="white")
        self.ax_plots.legend(facecolor="black", edgecolor="white")
        self.ax_plots.grid(True, linestyle="--", alpha=0.5)

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
            positions.append((x, y))
        return positions

    def connect_layers(self, ax, from_positions, to_positions):
        """
        Draw lines representing connections between two layers.

        Args:
            ax (matplotlib.axes.Axes): The axes on which to draw.
            from_positions (list): Positions of nodes in the preceding layer.
            to_positions (list): Positions of nodes in the next layer.

        Returns:
            None
        """
        for (x1, y1) in from_positions:
            for (x2, y2) in to_positions:
                ax.plot([x1, x2], [y1, y2], color='gray', linewidth=0.5)

    def start_training(self):
        """Starts the training process."""
        self.start_button.config(state="disabled")
        self.status_label.config(text="Training Started...")
        self.run_training_step()

    def run_training_step(self):
        """Runs one training step."""
        # Dummy logic to simulate training updates
        epoch_loss = np.random.uniform(0.1, 1.0)
        epoch_accuracy = np.random.uniform(70, 100)
        self.losses.append(epoch_loss)
        self.accuracies.append(epoch_accuracy)

        # Update plots
        self.loss_line.set_data(range(len(self.losses)), self.losses)
        self.accuracy_line.set_data(range(len(self.accuracies)), self.accuracies)
        self.ax_plots.relim()
        self.ax_plots.autoscale_view()
        self.network_canvas.draw()

        # Update status and metrics
        self.status_label.config(text=f"Epoch {self.current_epoch + 1} in progress...")
        self.metrics_label.config(text=f"Loss: {epoch_loss:.4f}\nAccuracy: {epoch_accuracy:.2f}%")
        self.current_epoch += 1

        # Schedule next epoch or finish training
        if self.current_epoch < self.epochs:
            self.master.after(1000, self.run_training_step)
        else:
            self.status_label.config(text="Training Complete!")
            self.start_button.config(state="normal")
