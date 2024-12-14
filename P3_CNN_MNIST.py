import numpy as np
import struct
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Class for convolutional layer
class Convolution:
    def __init__(self, input_shape, filter_size, num_filters):
        """
        Initializes a convolutional layer with the specified input shape, filter size, and number of filters.
        """
        self.input_channels, self.input_height, self.input_width = input_shape
        self.filter_size = filter_size
        self.num_filters = num_filters
        
        # Initialize weights using He initialization
        self.weights = np.random.randn(num_filters, self.input_channels, self.filter_size, filter_size) * np.sqrt(2 / (filter_size * filter_size * self.input_channels))
        self.biases = np.zeros(num_filters)

    def forward(self, input_data):
        """
        Performs the forward pass through the convolutional layer.
        """
        self.input_data = input_data
        batch_size, input_channels, input_height, input_width = input_data.shape
        self.output_height = input_height - self.filter_size + 1
        self.output_width = input_width - self.filter_size + 1

        # Extract patches using vectorized approach
        patches = np.zeros((batch_size, self.input_channels, self.filter_size, self.filter_size, self.output_height, self.output_width))
        for i in range(self.output_height):
            for j in range(self.output_width):
                patches[:, :, :, :, i, j] = input_data[:, :, i:i+self.filter_size, j:j+self.filter_size]

        # Reshape for matrix multiplication
        patches = patches.reshape(batch_size, self.input_channels * self.filter_size * self.filter_size, -1)
        weights = self.weights.reshape(self.num_filters, -1)

        # Matrix multiplication for convolution
        self.output = np.dot(weights, patches).reshape(batch_size, self.num_filters, self.output_height, self.output_width) + self.biases[:, np.newaxis, np.newaxis]
        return self.output

    def backward(self, dL_dout, learning_rate):
        """
        Performs the backward pass, computing gradients for weights and biases.
        """

        # Compute gradients and update weights and biases
        batch_size, num_filters, output_height, output_width = dL_dout.shape

        dL_dinput = np.zeros_like(self.input_data)
        dL_dweights = np.zeros_like(self.weights)
        dL_dbiases = np.zeros_like(self.biases)

        for n in range(batch_size):
            for f in range(num_filters):
                for i in range(output_height):
                    for j in range(output_width):
                        start_i, start_j = i, j
                        end_i, end_j = start_i + self.filter_size, start_j + self.filter_size

                        patch = self.input_data[n, :, start_i:end_i, start_j:end_j]
                        dL_dweights[f] += patch * dL_dout[n, f, i, j]
                        dL_dinput[n, :, start_i:end_i, start_j:end_j] += self.weights[f] * dL_dout[n, f, i, j]

                dL_dbiases[f] += np.sum(dL_dout[:, f, :, :])

        # Clipping is applied to prevent exploding gradients
        dL_dweights = clip_gradients(dL_dweights)
        dL_dinput = clip_gradients(dL_dinput)
        dL_dbiases = clip_gradients(dL_dbiases)

        # Update weights and biases
        self.weights -= learning_rate * dL_dweights / batch_size
        self.biases -= learning_rate * dL_dbiases / batch_size

        return dL_dinput

# Pooling layer class
class MaxPool:
    def __init__(self, pool_size):
        """
        Initializes a max-pooling layer with the specified pool size.
        """
        self.pool_size = pool_size

    def forward(self, input_data):
        """
        Performs the forward pass for max-pooling.
        """
        if input_data.ndim != 4:
            raise ValueError(f"Expected input_data to have 4 dimensions (batch_size, channels, height, width), but got {input_data.shape}")

        self.input_data = input_data
        batch_size, self.channels, self.input_height, self.input_width = input_data.shape
        self.output_height = self.input_height // self.pool_size
        self.output_width = self.input_width // self.pool_size
        
        self.output = np.zeros((batch_size, self.channels, self.output_height, self.output_width))

        for n in range(batch_size):
            for c in range(self.channels):
                for i in range(self.output_height):
                    for j in range(self.output_width):
                        start_i = i * self.pool_size
                        start_j = j * self.pool_size
                        end_i = start_i + self.pool_size
                        end_j = start_j + self.pool_size
                        patch = input_data[n, c, start_i:end_i, start_j:end_j]
                        self.output[n, c, i, j] = np.max(patch)

        return self.output

    def backward(self, dL_dout):
        """
        Performs the backward pass for max-pooling.
        """
        if dL_dout.ndim != 4:
            raise ValueError(f"Expected dL_dout to have 4 dimensions (batch_size, channels, output_height, output_width), but got {dL_dout.shape}")

        batch_size, channels, output_height, output_width = dL_dout.shape
        dL_dinput = np.zeros_like(self.input_data)

        for n in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        start_i = i * self.pool_size
                        start_j = j * self.pool_size
                        end_i = start_i + self.pool_size
                        end_j = start_j + self.pool_size
                        patch = self.input_data[n, c, start_i:end_i, start_j:end_j]
                        mask = patch == np.max(patch)
                        dL_dinput[n, c, start_i:end_i, start_j:end_j] = dL_dout[n, c, i, j] * mask

        return dL_dinput

class FullyConnected:
    
    def __init__(self, input_size, output_size):
        """
        Initializes a fully connected layer with specified input and output sizes.
        """
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
        self.biases = np.zeros(output_size)

    def forward(self, input_data):
        """
        Performs the forward pass through the dense layer.
        """
        self.input_data = input_data
        return np.dot(input_data, self.weights.T) + self.biases

    def softmax(self, z):
        exp_values = np.exp(z - np.max(z))
        return exp_values / np.sum(exp_values)

    def backward(self, dL_dout, lr):
        """
        Performs the backward pass, computing gradients for weights and biases.
        """
        batch_size = dL_dout.shape[0]

        dL_dw = np.dot(dL_dout.T, self.input_data) / batch_size
        dL_db = np.sum(dL_dout, axis=0) / batch_size

        dL_dw = clip_gradients(dL_dw)
        dL_db = clip_gradients(dL_db)

        dL_dinput = np.dot(dL_dout, self.weights) 
        dL_dinput = clip_gradients(dL_dinput) 

        self.weights -= lr * dL_dw
        self.biases -= lr * dL_db

        return dL_dinput

def cross_entropy_loss(predictions, labels):
    return -np.sum(labels * np.log(predictions + 1e-9)) / labels.shape[0]

def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

def load_mnist_images(filename):
    """
    Loads MNIST images from the given file.
    """
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols, 1)
        return images.astype(np.float32) / 255.0

def load_mnist_labels(filename):
    """
    Loads MNIST labels from the given file.
    """
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
        return labels
    
def relu(x):
    return np.maximum(0, x)

def softmax(z):
    exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

def l2_regularization(weights, lambda_):
    return lambda_ * np.sum(weights ** 2)

def clip_gradients(grad, threshold=1.0):
    norm = np.linalg.norm(grad)
    if norm > threshold:
        grad = grad * (threshold / norm)
    return grad

loss_values = []
stop_training = False

# Tkinter GUI for training and prediction
def create_ui():
    """
    Sets up the graphical user interface for the CNN MNIST classifier.
    """
    global canvas_nn, root, connection_lines, progress_bar, canvas_training_image, conv1, conv2
    def train_cnn():
        def training_process():
            global stop_training, X_train, y_train_one_hot, conv1, conv2, loss_values
            stop_training = False
            
            try:
                print("Reinitializing layers...")
                reset_training()
                conv1, conv2, fc, pool = initialize_layers()
                loss_values.clear()
                update_loss_plot()

                print("Layers reinitialized.")

                loss_values=[]

                # Reset progress bar
                progress_bar.update()
                root.after(0,update_architecture)
                
                n_samples = int(entry_samples.get())
                X_train = X_train[:n_samples]
                y_train_one_hot = y_train_one_hot[:n_samples]
                print(f"Using {n_samples} samples for training.")

                print("Training started...")
                
                epochs = int(entry_epochs.get())
                lr = float(entry_lr.get())
                batch_size = int(entry_batch_size.get())
                total_batches = len(X_train) // batch_size
                progress_bar['maximum'] = epochs

                for epoch in range(epochs):
                    if stop_training:
                        print("Training stopped")
                        root.after(0, result_text.set, "Training stopped")
                        return

                    print(f"Starting epoch {epoch + 1}/{epochs}...")
                    total_loss = 0
                    total_correct = 0

                    for i in range(total_batches):
                        if stop_training:
                            print("Training stopped")
                            root.after(0, result_text.set, "Training stopped")
                            return

                        batch_X = X_train[i * batch_size:(i + 1) * batch_size]
                        batch_y = y_train_one_hot[i * batch_size:(i + 1) * batch_size]

                        root.after(0, update_training_image, batch_X[0].squeeze())

                        # Forward pass
                        conv1_out = relu(conv1.forward(batch_X))
                        #print(f"conv1_out shape: {conv1_out.shape}")
                        conv2_out = relu(conv2.forward(conv1_out))
                        #print(f"conv2_out shape: {conv2_out.shape}")
                        #print(f"Shape after conv: {conv_out.shape}")
                        pool_out = pool.forward(conv2_out)
                        #print(f"pool_out shape: {pool_out.shape}")
                        #print(f"Shape after pool: {pool_out.shape}")
                        flattened_out = pool_out.reshape(pool_out.shape[0], -1)  # Forma: (batch_size, input_size)
                        #print(f"Shape after flatten: {flattened_out.shape}")

                        # Forward fully connected layer
                        predictions = softmax(fc.forward(flattened_out))  # Forma: (batch_size, output_size)
                            
                        # loss and accuracy
                        loss = cross_entropy_loss(predictions, batch_y)

                        # Regularization
                        lambda_ = 0.01 
                        l2_loss = l2_regularization(fc.weights, lambda_)
                        total_loss += loss + l2_loss

                        correct_predictions = np.sum(np.argmax(predictions,axis=1) == np.argmax(batch_y,axis=1))
                        total_correct += correct_predictions
                        
                        # Backward pass
                        grad = predictions - batch_y
                        fc_back = fc.backward(grad, lr)
                        #print(f"fc_back shape: {fc_back.shape}")
                        pool_back = pool.backward(fc_back.reshape(batch_X.shape[0], 32, 12, 12))
                        #print(f"pool_back shape: {pool_back.shape}")
                        conv2_back = conv2.backward(pool_back, lr)
                        #print(f"conv2_back shape: {conv2_back.shape}")
                        conv1.backward(conv2_back, lr)

                        root.after(0, update_architecture)

                        progress_bar['value'] = epoch + 1
                        root.update_idletasks()

                        #print(f"Gradients from fc_backward mean: {np.mean(fc_back)}")
                        #print(f"Gradients from pool_backward mean: {np.mean(pool_back)}")

                    avg_loss = total_loss / len(X_train)
                    accuracy = total_correct / len(X_train)
                    loss_values.append(avg_loss)
                    root.after(0,update_loss_plot)

                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")
                    root.after(0, result_text.set, f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")

                progress_bar['value'] = epochs
                root.update_idletasks()

                print("Training complete!")
                root.after(0, result_text.set, f"Training complete!\nEpoch:{epochs}\nLoss:{avg_loss:.4f}")

            except Exception as e:
                print(f"Error during training: {str(e)}")
                root.after(0, result_text.set, f"Error: {str(e)}")

        print("Starting thread...")
        training_thread = threading.Thread(target=training_process)
        training_thread.daemon = True
        training_thread.start()

    def on_close():
        global stop_training
        stop_training = True 
        print("Stopping training and closing application.")
        root.destroy()

    def predict_image():
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg")])
        if not file_path:
            return

        image = Image.open(file_path).convert('L').resize((28, 28))
        image_array = np.array(image) / 255.0

        image_array = image_array[np.newaxis, np.newaxis, :, :]

        # Forward pass
        conv1_out = relu(conv1.forward(image_array))  
        conv2_out = relu(conv2.forward(conv1_out))
        pool_out = pool.forward(conv2_out)
        flattened_out = pool_out.reshape(pool_out.shape[0], -1)
        predictions = softmax(fc.forward(flattened_out))

        predicted_class = np.argmax(predictions)

        img = ImageTk.PhotoImage(image.resize((100, 100)))
        canvas_image.create_image(50, 50, image=img, anchor=tk.CENTER)
        canvas_image.image = img

        result_text.set(f"Predicted Class: {predicted_class}")

    def update_loss_plot():
        #print(f"Current loss values: {loss_values}")
        loss_float_values = [float(val) for val in loss_values]

        loss_ax.clear()
        loss_ax.plot(range(1, len(loss_float_values) + 1), loss_float_values, marker="o", color="blue")
        loss_ax.set_xlim(1, int(entry_epochs.get())) 
        if loss_float_values:
            loss_ax.set_ylim(0, max(loss_float_values) * 1.1)
        else:
            loss_ax.set_ylim(0, 1)
        loss_ax.set_title("Loss per Epoch")
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Loss")
        loss_canvas.draw()  
    
    def update_training_image(image):
        global canvas_training_image
        image = (image * 255).astype(np.uint8) 
        pil_image = Image.fromarray(image.squeeze(), mode="L")  
        pil_image = pil_image.resize((200, 200))  
        tk_image = ImageTk.PhotoImage(pil_image)
        canvas_training_image.create_image(100, 100, image=tk_image, anchor=tk.CENTER)
        canvas_training_image.image = tk_image

    def predict_test_image(index):
        global X_test, y_test, conv1, conv2, fc, pool

        image_array = X_test[index:index+1]  # Get a single test image
        true_label = np.argmax(y_test[index])

        # Forward pass
        conv1_out = relu(conv1.forward(image_array))
        conv2_out = relu(conv2.forward(conv1_out))
        pool_out = pool.forward(conv2_out)
        flattened_out = pool_out.reshape(pool_out.shape[0], -1)
        predictions = softmax(fc.forward(flattened_out))

        predicted_class = np.argmax(predictions)

        # Display image and prediction
        img = ImageTk.PhotoImage(image=Image.fromarray((image_array[0, 0] * 255).astype(np.uint8)))
        canvas_image.create_image(50, 50, image=img, anchor=tk.CENTER)
        canvas_image.image = img

        result_text.set(f"True Class: {true_label}, Predicted Class: {predicted_class}")

    def reset_training():
        """
        Clears all training-related states to prepare for a new training session.
        """
        global loss_values, stop_training
        loss_values = []
        stop_training = False
        progress_bar['value'] = 0
        result_text.set("")

    root = tk.Tk()
    root.title("CNN MNIST Classifier")
    root.geometry("1200x700")

    # Frames
    left_frame = ttk.Frame(root, width=400, relief=tk.GROOVE, padding=10)
    left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

    right_frame = ttk.Frame(root, width=800, relief=tk.GROOVE, padding=10)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Matplotlib loss plot
    figure = Figure(figsize=(5, 2), dpi=100)
    loss_ax = figure.add_subplot(111)
    loss_ax.set_title("Loss per Epoch")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_canvas = FigureCanvasTkAgg(figure, master=right_frame)
    loss_canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    ttk.Label(left_frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W, pady=5)
    entry_epochs = ttk.Entry(left_frame)
    entry_epochs.insert(0, "5")
    entry_epochs.grid(row=0, column=1, pady=5)

    ttk.Label(left_frame, text="Learning Rate:").grid(row=1, column=0, sticky=tk.W, pady=5)
    entry_lr = ttk.Entry(left_frame)
    entry_lr.insert(0, "0.01")
    entry_lr.grid(row=1, column=1, pady=5)

    ttk.Label(left_frame, text="Batch Size:").grid(row=2, column=0, sticky=tk.W, pady=5)
    entry_batch_size = ttk.Entry(left_frame)
    entry_batch_size.insert(0, "64")
    entry_batch_size.grid(row=2, column=1, pady=5)

    ttk.Label(left_frame, text="Samples:").grid(row=3, column=0, sticky=tk.W, pady=5)
    entry_samples = ttk.Entry(left_frame)
    entry_samples.insert(0, "100")
    entry_samples.grid(row=3, column=1, pady=5)

    ttk.Button(left_frame, text="Train CNN", command=train_cnn).grid(row=4, columnspan=2, pady=10)
    ttk.Button(left_frame, text="Predict Image", command=predict_image).grid(row=10, columnspan=2, pady=10)

    # Add a button to evaluate test data
    ttk.Button(left_frame, text="Evaluate Test Set", command=evaluate_on_test_set).grid(row=11, columnspan=2, pady=10)

    # Results
    result_text = tk.StringVar()
    ttk.Label(left_frame, textvariable=result_text, font=("Arial", 12)).grid(row=6, columnspan=2, pady=10)

    # Progress bar
    progress_bar = ttk.Progressbar(right_frame, orient=tk.HORIZONTAL, length=700, mode='determinate')
    progress_bar.pack(side=tk.BOTTOM, pady=10)

    # Prediction canvas
    canvas_image = tk.Canvas(left_frame, width=200, height=200, bg="white")
    canvas_image.grid(row=9, columnspan=2, pady=10)

    # Canvas conf
    canvas_nn = tk.Canvas(right_frame, width=500, height=350, bg="white")
    canvas_nn.pack(fill=tk.BOTH, padx=10, pady=10)

    # Canvas for displaying training images
    canvas_training_image = tk.Canvas(left_frame, width=200, height=200, bg="white")
    canvas_training_image.grid(row=8, columnspan=2, pady=10)

    layers = [
        28 * 28,  # Inout layer
        16,  # conv1 output
        32,  # conv2 output
        10  # Final output
    ]
    
    def initialize_nn():
        global connection_lines, conv1, conv2
        layer_names = ["Input Layer", "Conv1", "Conv2", "Output Layer"]
        connection_lines = None
        reset_training()

    root.after(100, initialize_nn)
    root.mainloop()

def initialize_layers():
    global conv1, conv2, fc, pool
    """
    Initialize all layers and return them as a tuple.
    """
    print("Reinitializing layers...")
    conv1 = Convolution(input_shape=(1, 28, 28), filter_size=3, num_filters=16)
    conv2 = Convolution(input_shape=(16, 26, 26), filter_size=3, num_filters=32)
    input_size = 32 * 12 * 12  # Adjusted input size for fully connected layer
    fc = FullyConnected(input_size=input_size, output_size=10)
    pool = MaxPool(pool_size=2)
    print("Layers reinitialized.")
    return conv1, conv2, fc, pool

def update_architecture():
        global conv1, conv2
        """
        Updates the network architecture visualization in the UI
        """
        global conv1, conv2, fc, canvas_nn, connection_lines
        layers = [
            28 * 28,  # Input
            conv1.num_filters,  # conv1 output
            conv2.num_filters,  # conv2 output
            10  # Final output
        ]

        weights = [
            conv1.weights.reshape(-1, conv1.num_filters),
            conv2.weights.reshape(-1, conv2.num_filters),
            fc.weights
        ]
        
        connection_lines = draw_nn(canvas_nn, layers, weights, connection_lines)

def draw_nn(canvas_nn, layers, weights=None, connection_lines=None,layer_names=None):
    """
    Draw the NN in canvas
    """
    if connection_lines is None:
        connection_lines = {}
        canvas_nn.delete("all")
        layer_positions = []

        canvas_width = canvas_nn.winfo_width() or 600
        canvas_height = canvas_nn.winfo_height() or 400

        x_spacing = canvas_width / (len(layers) + 1)

        max_neurons = max(layers)
        max_visible_neurons = min(max_neurons, 10)
        y_spacing = canvas_height / (max_visible_neurons + 1)

        for i, layer_size in enumerate(layers):
            y_positions = []
            displayed_nodes = min(layer_size, max_visible_neurons)
            for j in range(displayed_nodes):
                x = (i + 1) * x_spacing
                y = (j + 1) * y_spacing
                y_positions.append((x, y))
                canvas_nn.create_oval(x - 10, y - 10, x + 10, y + 10, fill="lightblue", outline="black")
                if j == 0:
                    canvas_nn.create_text(x, 15, text=f"{layer_size} nodes", fill="black", font=("Arial", 8))

            if layer_names and i < len(layer_names):
                x = (i + 1) * x_spacing
                canvas_nn.create_text(x, y+20, text=layer_names[i], fill="black", font=("Arial", 8))

            layer_positions.append(y_positions)

        for i in range(len(layer_positions) - 1):
            prev_layer_positions = layer_positions[i]
            curr_layer_positions = layer_positions[i + 1]
            connection_lines[i] = []
            for j, prev_pos in enumerate(prev_layer_positions):
                for k, curr_pos in enumerate(curr_layer_positions):
                    line = canvas_nn.create_line(
                        prev_pos[0], prev_pos[1], curr_pos[0], curr_pos[1],
                        fill="gray", width=1
                    )
                    connection_lines[i].append((line, j, k))
        return connection_lines

    for i, layer_connections in connection_lines.items():
        if weights and i < len(weights):
            weight_matrix = weights[i]
            max_weight = np.max(np.abs(weight_matrix))
        else:
            weight_matrix = None
            max_weight = 1

        for line, j, k in layer_connections:
            if weight_matrix is not None and j < weight_matrix.shape[0] and k < weight_matrix.shape[1]:
                weight = weight_matrix[j, k]
                normalized_weight = abs(weight) / max_weight
                line_color = "red" if weight > 0 else "blue"
                line_width = 0.8 + normalized_weight * 3
                canvas_nn.itemconfig(line, fill=line_color, width=line_width)
    return connection_lines

def evaluate_on_test_set():
    global X_test, y_test, conv1, conv2, fc, pool
    
    total_correct = 0
    predictions = []
    for i in range(len(X_test)):
        # Forward pass through the network
        conv1_out = relu(conv1.forward(X_test[i:i+1]))  # Batch of 1
        conv2_out = relu(conv2.forward(conv1_out))
        pool_out = pool.forward(conv2_out)
        flattened_out = pool_out.reshape(pool_out.shape[0], -1)
        prediction = softmax(fc.forward(flattened_out))

        # Get the predicted class and check if correct
        predicted_class = np.argmax(prediction)
        true_class = np.argmax(y_test[i])
        predictions.append(predicted_class)
        if predicted_class == true_class:
            total_correct += 1
    
    accuracy = total_correct / len(X_test) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

def main():
    global conv, pool, fc, X_train, y_train_one_hot, conv1, conv2, X_test, y_test

    # Load MNIST data
    train_images_path = 'C:/Users/natal/OneDrive/Desktop/Selected topics/Corrections/CNN/archive (2)/train-images.idx3-ubyte'
    train_labels_path = 'C:/Users/natal/OneDrive/Desktop/Selected topics/Corrections/CNN/archive (2)/train-labels.idx1-ubyte'
    test_images_path = 'C:/Users/natal/OneDrive/Desktop/Selected topics/Corrections/CNN/archive (2)/t10k-images.idx3-ubyte'
    test_labels_path = 'C:/Users/natal/OneDrive/Desktop/Selected topics/Corrections/CNN/archive (2)/t10k-labels.idx1-ubyte'
    
    print("Loading MNIST data...")
    X_train = load_mnist_images(train_images_path)
    X_train = X_train.transpose(0, 3, 1, 2)  # (número, canales, altura, ancho)
    y_train = load_mnist_labels(train_labels_path)
    y_train_one_hot = one_hot_encode(y_train, 10)

    X_test = load_mnist_images(test_images_path).transpose(0, 3, 1, 2)  # Shape: (samples, channels, height, width)
    y_test = load_mnist_labels(test_labels_path)
    y_test = one_hot_encode(y_test, 10)

    print("MNIST data loaded successfully.")
    
    mean = np.mean(X_train, axis=(0, 2, 3), keepdims=True)
    std = np.std(X_train, axis=(0, 2, 3), keepdims=True)
    X_train = (X_train - mean) / (std + 1e-8) # normalization
    X_test = (X_test - mean) / (std + 1e-8)

    print("Starting UI...")

    create_ui()

if __name__ == "__main__":
    main()