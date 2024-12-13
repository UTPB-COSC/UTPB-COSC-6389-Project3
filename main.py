
from cnn_visualizer_ui import TrainerApp
from cnn_logic import load_dataset, Convolution, MaxPool, Fully_Connected
import tkinter as tk

if __name__ == "__main__":
    # Dummy dataset and model components for testing UI
    dataset_path = "dataset"
    X_train, X_test, y_train, y_test = load_dataset(dataset_path)

    conv = Convolution((64, 64), filter_size=3, num_filters=6)
    pool = MaxPool(pool_size=2)
    fully_connected_size = conv.num_filters * (64 // 2) * (64 // 2)
    full = Fully_Connected(input_size=fully_connected_size, output_size=2)

    # Initialize and run the TrainerApp
    root = tk.Tk()
    app = TrainerApp(root, X_train, y_train, conv, pool, full, lr=0.01, epochs=100)
    root.mainloop()
