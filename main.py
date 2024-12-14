import tkinter as tk
from tkinter import ttk
from src.layout import DiagramLayout
from src.dynamic import DiagramDynamic
from src.data import load_mnist_data
from src.network import Network
import numpy as np


def main():
    root = tk.Tk()
    root.title("Modular CNN Diagram with Training Controls")

    # Control frame at top
    control_frame = tk.Frame(root, padx=10, pady=10)
    control_frame.pack(side=tk.TOP, fill=tk.X)

    tk.Label(control_frame, text="Epochs:").grid(row=0, column=0, sticky="e")
    epochs_entry = tk.Entry(control_frame, width=5)
    epochs_entry.insert(0, "5")
    epochs_entry.grid(row=0, column=1)

    tk.Label(control_frame, text="Batch Size:").grid(row=0, column=2, sticky="e")
    batch_entry = tk.Entry(control_frame, width=5)
    batch_entry.insert(0, "64")
    batch_entry.grid(row=0, column=3)

    start_button = tk.Button(control_frame, text="Start Training")
    start_button.grid(row=1, column=0, pady=5)
    pause_button = tk.Button(control_frame, text="Pause")
    pause_button.grid(row=1, column=1)
    stop_button = tk.Button(control_frame, text="Stop")
    stop_button.grid(row=1, column=2)

    progress_label = tk.Label(control_frame, text="Progress: Not started")
    progress_label.grid(row=2, column=0, columnspan=4, sticky="w", pady=5)

    # Frame for canvas and scrollbars
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    hbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
    hbar.pack(side=tk.BOTTOM, fill=tk.X)

    vbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
    vbar.pack(side=tk.RIGHT, fill=tk.Y)

    canvas = tk.Canvas(
        frame,
        width=1500,
        height=800,
        bg="white",
        xscrollcommand=hbar.set,
        yscrollcommand=vbar.set,
    )
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    hbar.config(command=canvas.xview)
    vbar.config(command=canvas.yview)

    # Load data
    X_train, y_train, X_test, y_test = load_mnist_data()

    # Initialize network
    net = Network()
    net.forward(X_train[:1])  # set shapes

    layout = DiagramLayout(canvas)
    layout.draw_base_diagram()

    dynamic = DiagramDynamic(canvas, layout)

    # Training state in a dictionary
    training_state = {
        "is_training": False,
        "is_paused": False,
        "current_epoch": 0,
        "current_batch": 0,
        "max_epochs": 5,
        "batch_size": 64,
        "num_batches": 1,
        "current_index": 0,  # For cycling through X_train images for visualization
    }

    learning_rate = 0.01

    def update_progress():
        progress_label.config(
            text=f"Epoch {training_state['current_epoch']}/{training_state['max_epochs']}, Batch {training_state['current_batch']}/{training_state['num_batches']}"
        )

    def do_training_step():
        if not training_state["is_training"]:
            return

        if training_state["is_paused"]:
            root.after(500, do_training_step)
            return

        # Simulate one batch
        start = (training_state["current_batch"] - 1) * training_state["batch_size"]
        end = start + training_state["batch_size"]
        X_batch = X_train[start:end]
        y_batch = y_train[start:end]

        # Perform loss and backward pass
        loss = net.loss_and_backward(X_batch, y_batch)

        # Update params
        for layer in net.layers:
            if hasattr(layer, "W"):
                layer.W -= learning_rate * layer.dW
            if hasattr(layer, "b"):
                layer.b -= learning_rate * layer.db

        # Update progress
        update_progress()

        # Display a sample input from X_train itself as we cycle through them
        i = training_state["current_index"]
        img_4d = X_train[i : i + 1]
        label = y_train[i]

        # Convert to uint8 for display
        img_2d = (img_4d[0, 0] * 255).astype(np.uint8)
        dynamic.update_input_image(img_2d)

        # Forward with intermediates
        intermediates = net.forward_with_intermediates(img_4d)
        dynamic.update_values(intermediates, label)

        # Move to next image
        training_state["current_index"] = (
            training_state["current_index"] + 1
        ) % X_train.shape[0]

        training_state["current_batch"] += 1
        if training_state["current_batch"] > training_state["num_batches"]:
            training_state["current_batch"] = 1
            training_state["current_epoch"] += 1
            if training_state["current_epoch"] > training_state["max_epochs"]:
                training_state["is_training"] = False
                progress_label.config(text="Training Completed!")
                return

        # Schedule next batch step
        root.after(500, do_training_step)

    def start_training():
        try:
            training_state["max_epochs"] = int(epochs_entry.get())
            training_state["batch_size"] = int(batch_entry.get())
        except ValueError:
            training_state["max_epochs"] = 5
            training_state["batch_size"] = 64

        training_state["num_batches"] = X_train.shape[0] // training_state["batch_size"]
        training_state["current_epoch"] = 1
        training_state["current_batch"] = 1
        training_state["is_training"] = True
        training_state["is_paused"] = False
        update_progress()
        do_training_step()

    def pause_training():
        training_state["is_paused"] = not training_state["is_paused"]
        pause_button.config(text="Resume" if training_state["is_paused"] else "Pause")

    def stop_training():
        training_state["is_training"] = False
        progress_label.config(text="Training Stopped")

    start_button.config(command=start_training)
    pause_button.config(command=pause_training)
    stop_button.config(command=stop_training)

    root.after(1000, lambda: canvas.config(scrollregion=canvas.bbox("all")))

    root.mainloop()


if __name__ == "__main__":
    main()
