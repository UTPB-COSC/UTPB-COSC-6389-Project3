import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import numpy as np
from src.network import Network  # Ensure this matches your directory structure


class TrainingManager:
    def __init__(self):
        self.stop_requested = False
        self.pause_requested = False
        self.is_training = False
        self.current_epoch = 0
        self.current_batch = 0
        self.current_loss = 0.0
        self.max_epochs = 5
        self.batch_size = 64
        self.learning_rate = 0.01
        self.start_time = time.time()

        # New attributes for progress tracking
        self.num_batches = 0
        self.train_accuracies = []
        self.train_losses = []

    def request_stop(self):
        self.stop_requested = True

    def request_pause(self):
        self.pause_requested = True

    def resume(self):
        self.pause_requested = False


class DigitApp:
    def __init__(self, master):
        self.master = master
        master.title("Interactive CNN Training GUI")

        self.net = Network()
        self.training_manager = TrainingManager()

        # Main layout: Split into left stats panel and right main display
        main_frame = tk.Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left stats panel
        stats_frame = tk.Frame(main_frame, width=200, bg="#f0f0f0", padx=10, pady=10)
        stats_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Right main display frame
        display_frame = tk.Frame(main_frame, padx=10, pady=10)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Control Frame
        control_frame = tk.Frame(display_frame, padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Inputs for epochs, batch size, lr
        tk.Label(control_frame, text="Epochs:").grid(row=0, column=0, sticky="e")
        self.epochs_entry = tk.Entry(control_frame, width=5)
        self.epochs_entry.insert(0, "5")
        self.epochs_entry.grid(row=0, column=1)

        tk.Label(control_frame, text="Batch Size:").grid(row=0, column=2, sticky="e")
        self.batch_entry = tk.Entry(control_frame, width=5)
        self.batch_entry.insert(0, "64")
        self.batch_entry.grid(row=0, column=3)

        tk.Label(control_frame, text="Learning Rate:").grid(row=0, column=4, sticky="e")
        self.lr_entry = tk.Entry(control_frame, width=5)
        self.lr_entry.insert(0, "0.01")
        self.lr_entry.grid(row=0, column=5)

        # Buttons
        self.start_button = tk.Button(
            control_frame, text="Start", command=self.start_training
        )
        self.start_button.grid(row=1, column=0)

        self.pause_button = tk.Button(
            control_frame, text="Pause", command=self.pause_training
        )
        self.pause_button.grid(row=1, column=1)

        self.resume_button = tk.Button(
            control_frame, text="Resume", command=self.resume_training
        )
        self.resume_button.grid(row=1, column=2)

        self.stop_button = tk.Button(
            control_frame, text="Stop", command=self.stop_training
        )
        self.stop_button.grid(row=1, column=3)

        self.test_button = tk.Button(
            control_frame, text="Test Accuracy", command=self.test_accuracy
        )
        self.test_button.grid(row=1, column=4)

        # Notebook for tabs
        self.notebook = ttk.Notebook(display_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Training Status Tab
        self.status_frame = tk.Frame(self.notebook)
        self.notebook.add(self.status_frame, text="Training Status")

        self.epoch_label = tk.Label(self.status_frame, text="Epoch: 0/0")
        self.epoch_label.pack(anchor="w")

        self.epoch_progress = ttk.Progressbar(
            self.status_frame, length=200, mode="determinate"
        )
        self.epoch_progress.pack(anchor="w", pady=5)

        self.batch_label = tk.Label(self.status_frame, text="Batch: 0/0")
        self.batch_label.pack(anchor="w")

        self.batch_progress = ttk.Progressbar(
            self.status_frame, length=200, mode="determinate"
        )
        self.batch_progress.pack(anchor="w", pady=5)

        self.loss_label = tk.Label(self.status_frame, text="Loss: 0.00")
        self.loss_label.pack(anchor="w")

        self.accuracy_label = tk.Label(self.status_frame, text="Train Accuracy: N/A")
        self.accuracy_label.pack(anchor="w")

        # Network Info Tab
        self.arch_frame = tk.Frame(self.notebook)
        self.notebook.add(self.arch_frame, text="Network Info")

        self.arch_text = tk.Text(self.arch_frame, height=10, width=50)
        self.arch_text.pack(fill=tk.X, expand=False)

        self.arch_canvas = tk.Canvas(self.arch_frame, height=300, bg="white")
        self.arch_canvas.pack(fill=tk.BOTH, expand=True)

        self.weights_label = tk.Label(self.arch_frame, text="Weights Stats: N/A")
        self.weights_label.pack(anchor="w")

        # History/Log Tab
        self.history_frame = tk.Frame(self.notebook)
        self.notebook.add(self.history_frame, text="History/Log")

        self.history_text = tk.Text(self.history_frame, height=10, width=50)
        self.history_text.pack(fill=tk.BOTH, expand=True)

        # Side stats panel
        tk.Label(
            stats_frame,
            text="Real-Time Stats",
            bg="#f0f0f0",
            font=("Helvetica", 12, "bold"),
        ).pack(anchor="w", pady=5)
        self.stats_loss = tk.Label(stats_frame, text="Current Loss: N/A", bg="#f0f0f0")
        self.stats_loss.pack(anchor="w")
        self.stats_acc = tk.Label(
            stats_frame, text="Current Accuracy: N/A", bg="#f0f0f0"
        )
        self.stats_acc.pack(anchor="w")
        self.stats_lr = tk.Label(stats_frame, text="Learning Rate: 0.01", bg="#f0f0f0")
        self.stats_lr.pack(anchor="w")
        self.stats_time = tk.Label(stats_frame, text="Elapsed Time: 0s", bg="#f0f0f0")
        self.stats_time.pack(anchor="w")

        # DO NOT call self.update_architecture_display() here.
        # We'll call it after a dummy forward pass in main.py.

        # Update GUI status periodically
        self.update_gui_status()

    def update_architecture_display(self):
        self.arch_text.delete("1.0", tk.END)
        self.arch_canvas.delete("all")

        # Constants for drawing
        x_start = 50
        y_center = 150
        x_spacing = 150  # Brought layers closer
        feature_map_size = 20
        neuron_dot_radius = 5
        max_feature_maps_to_show = 8
        max_neurons_to_show = 20

        current_x = x_start
        previous_representation = None
        prev_is_conv = False

        def pick_repr_subset(coords):
            # Pick top, middle, and bottom coords to connect lines neatly if possible
            if len(coords) < 3:
                return coords
            return [coords[0], coords[len(coords) // 2], coords[-1]]

        def draw_conv_layer(layer, x_pos):
            out_c = getattr(layer, "out_channels", 8)
            out_h = getattr(layer, "out_height", 14)
            out_w = getattr(layer, "out_width", 14)

            show_c = min(out_c, max_feature_maps_to_show)
            scale_factor = feature_map_size / 28.0
            box_h = max(10, int(out_h * scale_factor))
            box_w = max(10, int(out_w * scale_factor))

            fm_x_start = x_pos
            fm_y_start = y_center - (show_c * (box_h + 5)) / 2
            coords = []
            for i in range(show_c):
                y_box = fm_y_start + i * (box_h + 5)
                self.arch_canvas.create_rectangle(
                    fm_x_start, y_box, fm_x_start + box_w, y_box + box_h, fill="#87CEEB"
                )
                coords.append((fm_x_start + box_w, y_box + box_h / 2))

            layer_info = (
                f"{type(layer).__name__}: {out_c} feature maps, {out_h}x{out_w}"
            )
            self.arch_text.insert(tk.END, layer_info + "\n")

            return coords

        def draw_fc_layer(layer, x_pos):
            out_dim = getattr(layer, "out_dim", 100)
            show_neurons = min(out_dim, max_neurons_to_show)

            start_x = x_pos
            dot_x = start_x
            dot_y = y_center
            coords = []
            for i in range(show_neurons):
                self.arch_canvas.create_oval(
                    dot_x,
                    dot_y - neuron_dot_radius,
                    dot_x + 2 * neuron_dot_radius,
                    dot_y + 2 * neuron_dot_radius,
                    fill="#90EE90",
                )
                coords.append((dot_x + neuron_dot_radius, dot_y))
                dot_x += 2 * neuron_dot_radius + 5

            layer_info = f"{type(layer).__name__}: {out_dim} neurons"
            self.arch_text.insert(tk.END, layer_info + "\n")

            return coords

        def draw_flatten_transition(prev_coords, x_pos):
            show_neurons = max_neurons_to_show
            dot_x = x_pos
            dot_y = y_center
            coords = []
            for i in range(show_neurons):
                self.arch_canvas.create_oval(
                    dot_x,
                    dot_y - neuron_dot_radius,
                    dot_x + 2 * neuron_dot_radius,
                    dot_y + 2 * neuron_dot_radius,
                    fill="#D3D3D3",
                )
                coords.append((dot_x + neuron_dot_radius, dot_y))
                dot_x += 2 * neuron_dot_radius + 5

            # Connect a few from prev_coords to these dots
            if prev_coords and coords:
                subset = pick_repr_subset(prev_coords)
                for px, py in subset:
                    self.arch_canvas.create_line(
                        px, py, coords[0][0], coords[0][1], fill="black"
                    )

            self.arch_text.insert(
                tk.END, "Flatten: converting feature maps to a vector\n"
            )
            return coords

        for i, layer in enumerate(self.net.layers):
            layer_type = type(layer).__name__
            x_pos = x_start + i * x_spacing

            if "Conv" in layer_type:
                coords = draw_conv_layer(layer, x_pos)
                if previous_representation and coords:
                    subset = pick_repr_subset(previous_representation)
                    for px, py in subset:
                        self.arch_canvas.create_line(
                            px, py, coords[0][0], coords[0][1], fill="black"
                        )
                previous_representation = coords
                prev_is_conv = True
            elif "FullyConnected" in layer_type:
                if prev_is_conv:
                    flatten_x = x_pos - 60
                    flat_coords = draw_flatten_transition(
                        previous_representation, flatten_x
                    )
                    previous_representation = flat_coords
                    prev_is_conv = False

                coords = draw_fc_layer(layer, x_pos)
                if previous_representation and coords:
                    subset = pick_repr_subset(previous_representation)
                    for px, py in subset:
                        self.arch_canvas.create_line(
                            px, py, coords[0][0], coords[0][1], fill="black"
                        )
                previous_representation = coords
            else:
                # Unknown layer type, just draw a simple box
                box_w = 80
                box_h = 40
                self.arch_canvas.create_rectangle(
                    x_pos,
                    y_center - box_h / 2,
                    x_pos + box_w,
                    y_center + box_h / 2,
                    fill="#e0e0e0",
                )
                coords = [(x_pos + box_w, y_center)]
                if previous_representation and coords:
                    subset = pick_repr_subset(previous_representation)
                    for px, py in subset:
                        self.arch_canvas.create_line(
                            px, py, coords[0][0], coords[0][1], fill="black"
                        )
                previous_representation = coords
                prev_is_conv = False
                self.arch_text.insert(tk.END, f"Layer {i}: {layer_type}\n")

        self.arch_text.insert(tk.END, "\nDiagram updated.\n")

    def start_training(self):
        try:
            self.training_manager.max_epochs = int(self.epochs_entry.get())
            self.training_manager.batch_size = int(self.batch_entry.get())
            self.training_manager.learning_rate = float(self.lr_entry.get())
        except ValueError:
            self.training_manager.max_epochs = 5
            self.training_manager.batch_size = 64
            self.training_manager.learning_rate = 0.01

        self.training_manager.stop_requested = False
        self.training_manager.pause_requested = False
        self.training_manager.is_training = True
        self.training_manager.current_epoch = 0
        self.training_manager.current_batch = 0
        self.training_manager.current_loss = 0.0
        self.training_manager.train_accuracies = []
        self.training_manager.train_losses = []
        self.training_manager.start_time = time.time()

        self.training_thread = threading.Thread(target=self.training_loop)
        self.training_thread.start()

    def stop_training(self):
        self.training_manager.request_stop()

    def pause_training(self):
        self.training_manager.request_pause()

    def resume_training(self):
        self.training_manager.resume()

    def test_accuracy(self):
        # Disable test button to prevent multiple clicks
        self.test_button.config(state=tk.DISABLED)
        test_thread = threading.Thread(target=self.compute_and_show_accuracy)
        test_thread.start()

    def compute_and_show_accuracy(self):
        from src.data import load_mnist_data

        X_train, y_train, X_test, y_test = load_mnist_data()
        acc = self.compute_accuracy(X_test, y_test)

        def show_result():
            self.test_button.config(state=tk.NORMAL)
            messagebox.showinfo("Test Accuracy", f"Accuracy: {acc * 100:.2f}%")

        self.master.after(0, show_result)

    def compute_accuracy(self, X, y, batch_size=100):
        N = X.shape[0]
        correct = 0
        for start_idx in range(0, N, batch_size):
            end_idx = start_idx + batch_size
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            out = self.net.forward(X_batch)
            preds = np.argmax(out, axis=1)
            correct += np.sum(preds == y_batch)
        return correct / N

    def training_loop(self):
        from src.data import load_mnist_data

        X_train, y_train, X_test, y_test = load_mnist_data()
        tm = self.training_manager
        tm.num_batches = X_train.shape[0] // tm.batch_size

        for epoch in range(1, tm.max_epochs + 1):
            if tm.stop_requested:
                break
            tm.current_epoch = epoch

            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            for i in range(tm.num_batches):
                if tm.stop_requested:
                    break
                while tm.pause_requested and not tm.stop_requested:
                    time.sleep(0.1)

                start = i * tm.batch_size
                end = start + tm.batch_size
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                loss = self.net.loss_and_backward(X_batch, y_batch)
                # Update parameters
                for layer in self.net.layers:
                    if hasattr(layer, "W"):
                        layer.W -= tm.learning_rate * layer.dW
                    if hasattr(layer, "b"):
                        layer.b -= tm.learning_rate * layer.db

                tm.current_batch = i + 1
                tm.current_loss = loss

            # After each epoch, compute training accuracy
            train_acc = self.compute_accuracy(X_train, y_train)
            tm.train_accuracies.append(train_acc)
            tm.train_losses.append(tm.current_loss)

            # Log to history
            self.history_text.insert(
                tk.END,
                f"Epoch {epoch}/{tm.max_epochs}: Loss={tm.current_loss:.4f}, Acc={train_acc*100:.2f}%\n",
            )
            self.history_text.see(tk.END)

            if tm.stop_requested:
                break

        tm.is_training = False

    def update_gui_status(self):
        tm = self.training_manager

        # Update epoch and batch indicators
        self.epoch_label.config(text=f"Epoch: {tm.current_epoch}/{tm.max_epochs}")
        self.batch_label.config(
            text=f"Batch: {tm.current_batch}/{tm.num_batches if tm.num_batches else 0}"
        )

        # Update progress bars
        if tm.max_epochs > 0:
            self.epoch_progress["maximum"] = tm.max_epochs
            self.epoch_progress["value"] = tm.current_epoch

        if tm.num_batches > 0:
            self.batch_progress["maximum"] = tm.num_batches
            self.batch_progress["value"] = tm.current_batch

        self.loss_label.config(text=f"Loss: {tm.current_loss:.4f}")

        # Update accuracy label
        if tm.train_accuracies:
            current_acc = tm.train_accuracies[-1] * 100
            self.accuracy_label.config(text=f"Train Accuracy: {current_acc:.2f}%")
        else:
            self.accuracy_label.config(text="Train Accuracy: N/A")

        # Update side panel stats
        self.stats_loss.config(text=f"Current Loss: {tm.current_loss:.4f}")
        if tm.train_accuracies:
            self.stats_acc.config(
                text=f"Current Accuracy: {tm.train_accuracies[-1]*100:.2f}%"
            )
        else:
            self.stats_acc.config(text="Current Accuracy: N/A")

        self.stats_lr.config(text=f"Learning Rate: {tm.learning_rate}")
        elapsed = int(time.time() - tm.start_time)
        self.stats_time.config(text=f"Elapsed Time: {elapsed}s")

        # Update weights stats if available
        if hasattr(self.net.layers[-1], "W"):
            W = self.net.layers[-1].W
            w_mean = np.mean(W)
            w_max = np.max(W)
            w_min = np.min(W)
            self.weights_label.config(
                text=f"Weights Stats: mean={w_mean:.4f}, max={w_max:.4f}, min={w_min:.4f}"
            )
        else:
            self.weights_label.config(text="Weights Stats: N/A")

        # Schedule next update
        self.master.after(500, self.update_gui_status)
