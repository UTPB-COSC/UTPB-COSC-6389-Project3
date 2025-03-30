import os
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import time  

IMAGE_SIZE = (64, 64)

# Step 1: Image Preprocessing
def preprocess_image(image_path):
    """Load and preprocess an image by resizing and normalizing."""
    image = Image.open(image_path).resize(IMAGE_SIZE).convert("RGB")
    data = np.array(image) / 255.0  
    return image, data

class SimpleCNN:
    def __init__(self, input_shape, num_classes, ui_callback=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weights = []  
        self.biases = []   
        self.ui_callback = ui_callback  
        self.build_network()

    def build_network(self):
        conv_filter_size = (3, 3)
        num_filters = 8
        self.weights.append(np.random.randn(*conv_filter_size, self.input_shape[2], num_filters) * 0.01)
        self.biases.append(np.zeros((num_filters,)))

        fc_input_size = (IMAGE_SIZE[0] - 2) * (IMAGE_SIZE[1] - 2) * num_filters
        self.weights.append(np.random.randn(fc_input_size, self.num_classes) * 0.01)
        self.biases.append(np.zeros((self.num_classes,)))

    def forward(self, x):
        conv_output = self.convolve(x, self.weights[0], self.biases[0])
        flattened = conv_output.flatten()
        output = np.dot(flattened, self.weights[1]) + self.biases[1]

        if self.ui_callback:
            self.ui_callback(self)  

        return output

    def convolve(self, x, filters, biases):
        filter_height, filter_width, _, num_filters = filters.shape
        input_height, input_width, _ = x.shape
        output_height = input_height - filter_height + 1
        output_width = input_width - filter_width + 1

        output = np.zeros((output_height, output_width, num_filters))

        for h in range(output_height):
            for w in range(output_width):
                for f in range(num_filters):
                    region = x[h:h + filter_height, w:w + filter_width, :]
                    output[h, w, f] = np.sum(region * filters[:, :, :, f]) + biases[f]

        return output

# Step 3: Visualization in the UI
class NeuralNetworkVisualizer(tk.Canvas):
    def __init__(self, parent, width, height):
        super().__init__(parent, width=width, height=height, bg="white")
        self.pack()

    def update_visualization(self, network):
        self.delete("all")
        x_start = 50
        y_start = 50
        layer_gap = 150
        neuron_gap = 50
        radius = 15

        for layer_idx, (weights, biases) in enumerate(zip(network.weights, network.biases)):
            x = x_start + layer_idx * layer_gap
            for neuron_idx, _ in enumerate(biases):
                y = y_start + neuron_idx * neuron_gap
                self.create_oval(x - radius, y - radius, x + radius, y + radius, fill="blue")

                if layer_idx > 0:  
                    prev_layer_neurons = len(network.biases[layer_idx - 1])
                    for prev_idx in range(prev_layer_neurons):
                        prev_x = x_start + (layer_idx - 1) * layer_gap
                        prev_y = y_start + prev_idx * neuron_gap
                        weight = weights[prev_idx, neuron_idx]
                        color = "green" if weight > 0 else "red"
                        self.create_line(prev_x + radius, prev_y, x - radius, y, fill=color)

        self.update_idletasks()
        self.update()

def load_dataset():
    folder_path = filedialog.askdirectory(title="Select Dataset Folder")
    if not folder_path:
        messagebox.showwarning("No Folder Selected", "Please select a folder to load the dataset.")
        return None

    fire_folder = os.path.join(folder_path, "fire")
    grass_folder = os.path.join(folder_path, "grass")

    if not os.path.exists(fire_folder) or not os.path.exists(grass_folder):
        messagebox.showerror("Invalid Dataset", "Dataset folder must contain 'fire' and 'grass' subfolders.")
        return None

    fire_images = [os.path.join(fire_folder, img) for img in os.listdir(fire_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    grass_images = [os.path.join(grass_folder, img) for img in os.listdir(grass_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

    return fire_images, grass_images

def run_classifier():
    dataset = load_dataset()
    if dataset is None:
        return

    fire_images, grass_images = dataset
    visualizer = NeuralNetworkVisualizer(side_frame, width=400, height=600)
    cnn = SimpleCNN(input_shape=(64, 64, 3), num_classes=2, ui_callback=visualizer.update_visualization)

    for img_path in fire_images + grass_images:
        original_image, image_data = preprocess_image(img_path)
        cnn.forward(image_data)

        img = ImageTk.PhotoImage(original_image)
        current_image_label.configure(image=img)
        current_image_label.image = img

        root.update()
        time.sleep(0.5)  

    messagebox.showinfo("Processing Complete", "All images have been processed.")
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Pokemon Type Classifier")

    left_frame = tk.Frame(root)
    left_frame.pack(side=tk.LEFT, padx=10, pady=10)

    current_image_label = tk.Label(left_frame)
    current_image_label.pack(pady=10)

    side_frame = tk.Frame(root)
    side_frame.pack(side=tk.RIGHT, padx=10, pady=10)

    load_button = tk.Button(root, text="Load Dataset and Classify", command=run_classifier)
    load_button.pack(pady=20)

    root.mainloop()
