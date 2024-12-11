import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from cnn import SimpleCNN, softmax, softmax_derivative
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import os


# Helper Function to Load Dataset
def load_training_data(csv_file, image_folder, image_size=(64, 64)):
    data = []
    labels = []
    df = pd.read_csv(csv_file)
    class_names = sorted(df['label'].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    for _, row in df.iterrows():
        try:
            img_path = os.path.join(image_folder, row['image_name'])
            img = Image.open(img_path).convert('RGB')
            img = img.resize(image_size)
            data.append(np.array(img) / 255.0)  # Normalize
            labels.append(class_to_idx[row['label']])
        except Exception as e:
            print(f"Error loading image {row['image_name']}: {e}")

    data = np.array(data)
    labels_one_hot = np.zeros((len(labels), len(class_names)))
    labels_one_hot[np.arange(len(labels)), labels] = 1

    return data, labels_one_hot, class_names


# CNNVisualizer Class
class CNNVisualizer:
    def __init__(self, root, cnn_model, data_train, labels_train, class_names):
        self.root = root
        self.cnn_model = cnn_model
        self.data_train = data_train
        self.labels_train = labels_train
        self.class_names = class_names
        self.losses = []
        self.accuracies = []

        # UI Layout: Split into Graph, Network, and Image Visualizations
        self.network_canvas = tk.Canvas(root, width=400, height=600, bg="white")
        self.network_canvas.grid(row=0, column=0, rowspan=2, padx=10, pady=10)

        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.ax_loss = self.figure.add_subplot(211)
        self.ax_accuracy = self.figure.add_subplot(212)
        self.figure.subplots_adjust(hspace=0.8)

        self.graph_canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.graph_canvas.get_tk_widget().grid(row=0, column=1, padx=10, pady=10)

        # Label to show exact accuracy and loss
        self.info_label = tk.Label(root, text="Loss: N/A, Accuracy: N/A", font=("Arial", 14))
        self.info_label.grid(row=1, column=1, padx=10, pady=10)

        # Predicted and True Image Canvases
        self.predicted_image_canvas = tk.Canvas(root, width=150, height=150, bg="white", highlightthickness=1)
        self.predicted_image_canvas.grid(row=2, column=0, padx=10, pady=10)
        self.predicted_label = tk.Label(root, text="Predicted Image", font=("Arial", 12))
        self.predicted_label.grid(row=3, column=0)

        self.true_image_canvas = tk.Canvas(root, width=150, height=150, bg="white", highlightthickness=1)
        self.true_image_canvas.grid(row=2, column=1, padx=10, pady=10)
        self.true_label = tk.Label(root, text="True Image", font=("Arial", 12))
        self.true_label.grid(row=3, column=1)

        self.start_button = tk.Button(root, text="Start Training", command=self.start_training)
        self.start_button.grid(row=4, column=0, columnspan=2, pady=10)

        self.tk_predicted_image = None
        self.tk_true_image = None

    def draw_network(self):
        """Draw a neural network visualization with dynamic connection colors."""
        self.network_canvas.delete("all")
        # Positions and Neurons Per Layer
        layer_positions = [100, 200, 300]
        neurons_per_layer = [5, 10, len(self.class_names)]
        layer_colors = ["lightblue", "lightgreen", "lightcoral"]

        connection_weights = [
            self.cnn_model.fc1.weights,
            self.cnn_model.fc2.weights,
        ]

        def get_color(weight):
            intensity = max(0, min(255, int((weight + 1) * 127.5)))
            return f"#{intensity:02x}{intensity:02x}ff"

        for layer_index, x in enumerate(layer_positions):
            y_positions = np.linspace(50, 550, neurons_per_layer[layer_index])
            for y in y_positions:
                self.network_canvas.create_oval(
                    x - 10, y - 10, x + 10, y + 10, fill=layer_colors[layer_index], outline="black"
                )

            if layer_index < len(layer_positions) - 1:
                next_y_positions = np.linspace(50, 550, neurons_per_layer[layer_index + 1])
                weights = connection_weights[layer_index]
                for i, y1 in enumerate(y_positions):
                    for j, y2 in enumerate(next_y_positions):
                        color = get_color(weights[i, j])
                        self.network_canvas.create_line(
                            x + 10, y1, layer_positions[layer_index + 1] - 10, y2, fill=color
                        )

    def update_graph(self, epoch, loss, accuracy):
        """Update the graphs for loss and accuracy."""
        self.losses.append(loss)
        self.accuracies.append(accuracy)

        self.ax_loss.clear()
        self.ax_loss.plot(self.losses, label="Loss", color="blue")
        self.ax_loss.set_title("Training Loss")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.legend()

        self.ax_accuracy.clear()
        self.ax_accuracy.plot(self.accuracies, label="Accuracy", color="green")
        self.ax_accuracy.set_title("Training Accuracy")
        self.ax_accuracy.set_xlabel("Epoch")
        self.ax_accuracy.set_ylabel("Accuracy")
        self.ax_accuracy.legend()

        self.graph_canvas.draw()
        self.info_label.config(text=f"Epoch: {epoch} | Loss: {loss:.4f} | Accuracy: {accuracy:.2f}%")

    def display_images(self, predicted_image, true_image):
        """Display the predicted and true images on the UI."""
        predicted_pil_image = Image.fromarray((predicted_image * 255).astype(np.uint8))
        true_pil_image = Image.fromarray((true_image * 255).astype(np.uint8))

        predicted_pil_image = predicted_pil_image.resize((150, 150))
        true_pil_image = true_pil_image.resize((150, 150))

        self.tk_predicted_image = ImageTk.PhotoImage(predicted_pil_image)
        self.tk_true_image = ImageTk.PhotoImage(true_pil_image)

        self.predicted_image_canvas.create_image(75, 75, image=self.tk_predicted_image)
        self.true_image_canvas.create_image(75, 75, image=self.tk_true_image)

    def start_training(self):
        """Train the CNN model on the loaded dataset."""
        self.losses.clear()
        self.accuracies.clear()

        for epoch in range(1, 101):
            example_image = self.data_train[epoch % len(self.data_train)]
            target = self.labels_train[epoch % len(self.labels_train)]

            output = softmax(self.cnn_model.forward(example_image[np.newaxis, :]))
            loss = -np.sum(target * np.log(output))

            grad = softmax_derivative(output, target[np.newaxis, :])
            self.cnn_model.backward(grad)
            self.cnn_model.update(learning_rate=0.01)

            predicted_class = np.argmax(output)
            true_class = np.argmax(target)
            accuracy = 100.0 if predicted_class == true_class else 0.0

            predicted_image = self.data_train[predicted_class]
            true_image = example_image

            self.display_images(predicted_image, true_image)
            self.draw_network()
            self.update_graph(epoch, loss, accuracy)

            self.root.update()
            self.root.after(500)


# Main Entry
if __name__ == "__main__":
    train_csv = "./dataset/Training_set.csv"
    train_folder = "./dataset/train"
    data_train, labels_train, class_names = load_training_data(train_csv, train_folder)

    root = tk.Tk()
    root.title("CNN Training Visualization")
    cnn = SimpleCNN(input_shape=(64, 64, 3), num_classes=len(class_names))

    visualizer = CNNVisualizer(root, cnn, data_train, labels_train, class_names)
    root.mainloop()
