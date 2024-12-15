# Download: https://www.kaggle.com/datasets/avnishnish/mnist-original

# Imported packages
import numpy as np
import tkinter as tk
import threading
import matplotlib
import matplotlib.pyplot as plt
import random
from tkinter import ttk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix

# We use this so that matplotlib doesn't try to open its own window
matplotlib.use("TkAgg")

# ---------------------------------------------
# Utility / Helper functions
# ---------------------------------------------

def clip_gradients(grad, threshold=1.0):
    """
    Clips gradients if their norm exceeds a given threshold to avoid exploding gradients.
    """
    norm = np.linalg.norm(grad)
    if norm > threshold:
        grad = grad * (threshold / norm)
    return grad

def one_hot_encode(labels, num_classes):
    """
    Converts integer labels (0..9) to one-hot encoded labels
    of shape (num_samples, num_classes), e.g. 3 -> [0,0,0,1,0,0,0,0,0,0].
    """
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

def cross_entropy_loss(predictions, labels):
    """
    Computes cross-entropy loss between softmax predictions and one-hot labels.
    predictions: shape (batch_size, num_classes), each row sums to 1
    labels: shape (batch_size, num_classes), 1-hot
    """
    return -np.sum(labels * np.log(predictions + 1e-9)) / labels.shape[0]

def relu(x):
    """
    ReLU activation function: max(0, x)
    """
    return np.maximum(0, x)

def d_relu(x):
    """
    Derivative of ReLU: 1 if x>0, else 0
    """
    return (x > 0).astype(x.dtype)

def softmax(z):
    """
    Softmax converts logits into probabilities that sum to 1 across each row.
    """
    exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

def l2_regularization(weights, lambda_):
    """
    Computes L2 regularization term for the weights to help reduce overfitting.
    """
    return lambda_ * np.sum(weights ** 2)

# ------------------------------------------------
# im2col / col2im for vectorized Convolution
# ------------------------------------------------

def im2col(inputs, filter_height, filter_width, stride=1, pad=0):
    """
    Reshapes image patches into columns for vectorized convolution.
    This technique is widely used in many CNN frameworks for speed.
    """
    batch_size, channels, height, width = inputs.shape
    out_height = (height + 2 * pad - filter_height) // stride + 1
    out_width = (width + 2 * pad - filter_width) // stride + 1

    # Pad the inputs with zeros if necessary
    padded = np.pad(inputs, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    # The output shape for columns
    cols = np.zeros((channels * filter_height * filter_width, out_height * out_width * batch_size))

    col_index = 0
    # Slide over the input spatially and flatten patches
    for y in range(out_height):
        for x in range(out_width):
            patch = padded[:, :, y*stride:y*stride+filter_height, x*stride:x*stride+filter_width]
            # patch has shape (batch_size, channels*filter_height*filter_width)
            patch = patch.reshape(batch_size, -1)
            cols[:, col_index:col_index + batch_size] = patch.T
            col_index += batch_size
    return cols

def col2im(cols, batch_size, channels, height, width, filter_height, filter_width,
           stride=1, pad=0):
    """
    Reverts the columns (flattened patches) back into the original
    spatial arrangement during the backward pass of convolution.
    """
    out_height = (height + 2 * pad - filter_height) // stride + 1
    out_width = (width + 2 * pad - filter_width) // stride + 1
    padded = np.zeros((batch_size, channels, height + 2 * pad, width + 2 * pad))

    col_index = 0
    # Inverse of the im2col
    for y in range(out_height):
        for x in range(out_width):
            patch = cols[:, col_index:col_index + batch_size].T
            patch = patch.reshape(batch_size, channels, filter_height, filter_width)
            padded[:, :, y*stride:y*stride+filter_height, x*stride:x*stride+filter_width] += patch
            col_index += batch_size

    # Remove padding if pad > 0
    if pad == 0:
        return padded
    return padded[:, :, pad:-pad, pad:-pad]


# ---------------------------
# Adam Optimizer Class
# ---------------------------

class AdamOptimizer:
    """
    Implements the Adam optimization algorithm with bias-correction.
    """
    def __init__(self, shape_w, shape_b, lr=0.001, beta1=0.9, beta2=0.999):
        # lr = learning rate, beta1/beta2 = exponential decay rates
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = 1e-7
        self.t = 0  # iteration count

        # Initialize moment estimates
        self.m_w = np.zeros(shape_w)
        self.v_w = np.zeros(shape_w)
        self.m_b = np.zeros(shape_b)
        self.v_b = np.zeros(shape_b)

    def update(self, w, b, dw, db):
        """
        Update parameters w, b given gradients dw, db using Adam.
        Includes gradient clipping to avoid exploding gradients.
        """
        self.t += 1
        dw = clip_gradients(dw)
        db = clip_gradients(db)

        # Exponential moving averages of gradient and grad^2
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dw
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (dw ** 2)
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db ** 2)

        # Bias correction
        m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
        v_w_hat = self.v_w / (1 - self.beta2 ** self.t)
        w -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.eps)

        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)
        b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)

        return w, b


# ---------------------------
# Convolution Layer Class
# ---------------------------

class Convolution:
    """
    A 2D convolution layer using im2col/col2im for vectorized forward/backward passes.
    """
    def __init__(self, input_shape, filter_size, num_filters, stride=1, pad=0, lr=0.001):
        # input_shape is (channels, height, width)
        self.input_channels, self.input_height, self.input_width = input_shape
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.stride = stride
        self.pad = pad

        # He initialization (Kaiming)
        self.weights = (np.random.randn(num_filters, self.input_channels, filter_size, filter_size)
                        * np.sqrt(2.0 / (self.input_channels * filter_size * filter_size)))
        self.biases = np.zeros(num_filters)

        # Adam optimizer for this layer
        self.optimizer = AdamOptimizer(shape_w=self.weights.shape, shape_b=self.biases.shape, lr=lr)

    def forward(self, input_data):
        """
        Forward pass. We reshape the patches using im2col and multiply by the filter weights.
        """
        self.input_data = input_data
        self.batch_size, _, _, _ = input_data.shape

        # Calculate output height and width
        self.out_height = (self.input_height + 2*self.pad - self.filter_size)//self.stride + 1
        self.out_width = (self.input_width + 2*self.pad - self.filter_size)//self.stride + 1

        if self.out_height <= 0 or self.out_width <= 0:
            raise ValueError(
                f"Invalid Conv layer parameters: input ({self.input_height}, {self.input_width}), "
                f"filter size ({self.filter_size}), stride ({self.stride}), pad ({self.pad}). "
                f"Got output dims: ({self.out_height}, {self.out_width})."
            )

        # Flatten the filter weights so shape: (num_filters, channels*filter_size^2)
        self.W_col = self.weights.reshape(self.num_filters, -1)
        # Flatten input patches via im2col
        self.X_col = im2col(input_data, self.filter_size, self.filter_size,
                            stride=self.stride, pad=self.pad)

        # Multiply W_col by X_col and add biases
        out_col = self.W_col @ self.X_col + self.biases.reshape(-1, 1)
        out = out_col.reshape(self.num_filters, self.out_height, self.out_width, self.batch_size)

        # Reorder to (batch, filters, out_height, out_width)
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, dL_dout):
        """
        Backprop through convolution using col2im to revert flattened patches.
        """
        batch_size, num_filters, out_h, out_w = dL_dout.shape

        # Reshape dL_dout to match out_col shape
        dL_dout_reshaped = dL_dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)

        # Gradient wrt weights
        dW = dL_dout_reshaped @ self.X_col.T
        dW = dW.reshape(self.weights.shape)

        # Gradient wrt biases
        dB = np.sum(dL_dout_reshaped, axis=1)

        # Gradient wrt input data
        W_col_T = self.W_col.T
        dX_col = W_col_T @ dL_dout_reshaped
        dX = col2im(dX_col, batch_size, self.input_channels, self.input_height, self.input_width,
                    self.filter_size, self.filter_size, stride=self.stride, pad=self.pad)

        # Update parameters
        self.weights, self.biases = self.optimizer.update(self.weights, self.biases, dW, dB)
        return dX


# ---------------------------
# Max Pooling Layer Class
# ---------------------------

class MaxPool:
    """
    Max Pooling layer to reduce spatial dimensions, picking the maximum value in each window.
    """
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input_data):
        """
        Forward pass picks max from each non-overlapping region of size (pool_size x pool_size).
        """
        self.input_data = input_data
        batch_size, channels, in_h, in_w = input_data.shape

        out_h = (in_h - self.pool_size) // self.stride + 1
        out_w = (in_w - self.pool_size) // self.stride + 1

        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                f"Invalid pooling params: input shape ({in_h}, {in_w}), "
                f"pool size ({self.pool_size}), stride ({self.stride})."
            )

        self.output = np.zeros((batch_size, channels, out_h, out_w))
        self.mask = np.zeros_like(input_data)  # will store the max location

        # For each window in each channel, pick the max
        for n in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        r0 = i * self.stride
                        c0 = j * self.stride
                        r1 = r0 + self.pool_size
                        c1 = c0 + self.pool_size
                        patch = input_data[n, c, r0:r1, c0:c1]
                        max_val = np.max(patch)
                        self.output[n, c, i, j] = max_val
                        self.mask[n, c, r0:r1, c0:c1] = (patch == max_val)
        return self.output

    def backward(self, dL_dout):
        """
        Backprop: pass the gradient to whichever element was the max in each region.
        """
        batch_size, channels, out_h, out_w = dL_dout.shape
        dL_dinput = np.zeros_like(self.input_data)

        for n in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        grad = dL_dout[n, c, i, j]
                        r0 = i * self.stride
                        c0 = j * self.stride
                        r1 = r0 + self.pool_size
                        c1 = c0 + self.pool_size
                        dL_dinput[n, c, r0:r1, c0:c1] += self.mask[n, c, r0:r1, c0:c1] * grad
        return dL_dinput


# ---------------------------------
# Fully Connected (Dense) Layer
# ---------------------------------

class FullyConnected:
    """
    A Fully Connected layer (dense) for classification or final output in the CNN.
    """
    def __init__(self, input_size, output_size, lr=0.0005):
        # He initialization
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros(output_size)
        self.optimizer = AdamOptimizer(self.weights.shape, self.biases.shape, lr=lr)

    def forward(self, input_data):
        """
        Forward pass: X * W^T + b
        input_data shape: (batch_size, input_size)
        """
        self.input_data = input_data
        return np.dot(input_data, self.weights.T) + self.biases

    def backward(self, dL_dout):
        """
        Backward pass: compute dW, dB, and pass gradient to the previous layer.
        """
        batch_size = dL_dout.shape[0]
        dW = np.dot(dL_dout.T, self.input_data) / batch_size
        dB = np.sum(dL_dout, axis=0) / batch_size
        dInput = np.dot(dL_dout, self.weights)
        # Update parameters with Adam
        self.weights, self.biases = self.optimizer.update(self.weights, self.biases, dW, dB)
        return dInput


# Globals for storing data, layers, etc.
accuracy_values = []
stop_training = False
X_train = None
y_train_one_hot = None
X_test = None
y_test = None

conv1 = None
pool1 = None
conv2 = None
pool2 = None
fc = None

confusion_ax = None
confusion_canvas = None


# ----------------------------------------------
# create_ui() builds the Tkinter Graphical UI
# ----------------------------------------------
def create_ui():
    """
    Creates the main GUI for building and training the CNN on MNIST data.
    """
    global root, progress_bar, canvas_training_image
    global entry_epochs, entry_lr, entry_conv1_size, entry_conv2_size
    global entry_pool_size1, entry_pool_size2
    global canvas_nn, result_text, acc_ax, acc_canvas
    global conv1, conv2, pool1, pool2, fc
    global accuracy_values
    global confusion_ax, confusion_canvas

    # Default training parameters
    BATCH_SIZE = 64
    SAMPLES = 200

    def train_cnn():
        """
        Starts CNN training in a separate thread, preventing UI freeze.
        """
        def training_process():
            global stop_training, X_train, y_train_one_hot, conv1, conv2, pool1, pool2, fc, accuracy_values
            stop_training = False
            try:
                reset_training()
                conv1, pool1, conv2, pool2, fc = initialize_layers()
                root.after(0, update_architecture)

                accuracy_values.clear()
                update_accuracy_plot()

                n_samples = SAMPLES
                epochs = int(entry_epochs.get())
                batch_size = BATCH_SIZE

                # Subset of the training data for quick experimentation
                X_sub = X_train[:n_samples]
                y_sub = y_train_one_hot[:n_samples]
                total_batches = len(X_sub) // batch_size

                progress_bar['maximum'] = epochs

                for epoch in range(epochs):
                    if stop_training:
                        root.after(0, result_text.set, "Training stopped")
                        return

                    # Shuffle the subset every epoch
                    indices = np.arange(len(X_sub))
                    np.random.shuffle(indices)
                    X_sub = X_sub[indices]
                    y_sub = y_sub[indices]

                    total_correct = 0
                    for i in range(total_batches):
                        if stop_training:
                            root.after(0, result_text.set, "Training stopped")
                            return

                        # Extract mini-batch
                        batch_X = X_sub[i * batch_size:(i + 1) * batch_size]
                        batch_y = y_sub[i * batch_size:(i + 1) * batch_size]

                        # Show the first image from the batch on the left-frame
                        root.after(0, update_training_image, batch_X[0].squeeze())

                        # Forward pass
                        out_c1 = relu(conv1.forward(batch_X))
                        out_p1 = pool1.forward(out_c1)
                        out_c2 = relu(conv2.forward(out_p1))
                        out_p2 = pool2.forward(out_c2)
                        flattened = out_p2.reshape(out_p2.shape[0], -1)
                        logits = fc.forward(flattened)
                        predictions = softmax(logits)

                        # Check how many predictions matched
                        correct_predictions = np.sum(
                            np.argmax(predictions, axis=1) == np.argmax(batch_y, axis=1)
                        )
                        total_correct += correct_predictions

                        # Backpropagation
                        grad = (predictions - batch_y)
                        grad_fc = fc.backward(grad)
                        grad_fc = grad_fc.reshape(out_p2.shape)
                        grad_p2 = pool2.backward(grad_fc)
                        grad_c2 = grad_p2 * d_relu(out_c2)
                        grad_c2 = conv2.backward(grad_c2)
                        grad_p1 = pool1.backward(grad_c2)
                        grad_c1 = grad_p1 * d_relu(out_c1)
                        conv1.backward(grad_c1)

                    # Accuracy for this epoch
                    accuracy = total_correct / len(X_sub)
                    accuracy_values.append(accuracy)
                    root.after(0, update_accuracy_plot)

                    progress_bar['value'] = epoch + 1
                    root.update_idletasks()

                    print(f"Epoch {epoch + 1}/{epochs} | Accuracy: {accuracy:.2%}")
                    root.after(0, result_text.set,
                               f"Epoch {epoch + 1}/{epochs} | Accuracy: {accuracy:.2%}")

                # Done training
                print("Training complete!")
                root.after(0, result_text.set,
                           f"Training complete! Final Accuracy: {accuracy:.2%}")

            except Exception as e:
                print(f"Error during training: {str(e)}")
                root.after(0, result_text.set, f"Error: {str(e)}")

        # Start the separate thread for training
        training_thread = threading.Thread(target=training_process)
        training_thread.daemon = True
        training_thread.start()

    def on_close():
        """
        Closes the UI and stops any ongoing training.
        """
        global stop_training
        stop_training = True
        root.destroy()

    def predict_dataset_sample():
        """
        Picks a random image from X_test, forward passes it through the CNN,
        and displays the predicted class and the original image on "Predicted Image" canvas.
        """
        if X_test is None or y_test is None:
            result_text.set("No dataset loaded.")
            return

        idx = random.randint(0, len(X_test) - 1)
        sample = X_test[idx:idx+1]  # shape (1,1,28,28)
        true_label = np.argmax(y_test[idx])

        # Forward pass on this single sample
        out_c1 = relu(conv1.forward(sample))
        out_p1 = pool1.forward(out_c1)
        out_c2 = relu(conv2.forward(out_p1))
        out_p2 = pool2.forward(out_c2)
        flattened = out_p2.reshape(out_p2.shape[0], -1)
        logits = fc.forward(flattened)
        predictions = softmax(logits)

        predicted_class = np.argmax(predictions)

        # Display info
        result_text.set(f"Dataset Index: {idx}\nPredicted: {predicted_class} | True: {true_label}")

        # Show the predicted image (from test set) on "Predicted Image" canvas
        update_predicted_image(sample[0].squeeze())  # shape (28,28)

    def evaluate_on_test_set():
        """
        Evaluates the CNN on the entire test set. Displays confusion matrix & accuracy.
        """
        global confusion_ax, confusion_canvas
        total_correct = 0
        all_preds = []
        all_true = []
        for i in range(len(X_test)):
            out_c1 = relu(conv1.forward(X_test[i:i + 1]))
            out_p1 = pool1.forward(out_c1)
            out_c2 = relu(conv2.forward(out_p1))
            out_p2 = pool2.forward(out_c2)
            flattened = out_p2.reshape(out_p2.shape[0], -1)
            logits = fc.forward(flattened)
            predictions = softmax(logits)
            predicted_class = np.argmax(predictions)
            true_class = np.argmax(y_test[i])

            all_preds.append(predicted_class)
            all_true.append(true_class)
            if predicted_class == true_class:
                total_correct += 1

        accuracy = total_correct / len(X_test) * 100
        print(f"Test Accuracy: {accuracy:.2f}%")
        result_text.set(f"Test Accuracy: {accuracy:.2f}%")

        # Plot the confusion matrix
        cm = confusion_matrix(all_true, all_preds, labels=range(10))
        confusion_ax.clear()
        confusion_ax.set_title("Testing Confusion Matrix")
        im = confusion_ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        confusion_ax.figure.colorbar(im, ax=confusion_ax)
        confusion_canvas.draw()

    def reset_training():
        """
        Clears accuracy values and resets the training progress bar and status text.
        """
        global accuracy_values, stop_training
        accuracy_values = []
        stop_training = False
        progress_bar['value'] = 0
        result_text.set("")

    def update_accuracy_plot():
        """
        Draw the training accuracy line plot on the Tkinter canvas each epoch.
        """
        acc_ax.clear()
        acc_ax.set_title("Training Accuracy")
        acc_ax.set_xlabel("Epochs")
        acc_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if len(accuracy_values) > 0:
            acc_ax.plot(range(1, len(accuracy_values) + 1), accuracy_values,
                        marker='o', color="blue")
            acc_max = max(accuracy_values)
            acc_ax.set_ylim(0, acc_max * 1.1 if acc_max < 1.0 else 1.1)
        else:
            acc_ax.set_ylim(0, 1)
        acc_canvas.draw()

    def update_training_image(image):
        """
        Displays a single training sample (shape 28x28) on the left panel
        so user can see what the CNN is training on.
        """
        image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image, mode="L")
        pil_image = pil_image.resize((100, 100))
        tk_image = ImageTk.PhotoImage(pil_image)
        canvas_training_image.create_image(50, 50, image=tk_image, anchor=tk.CENTER)
        canvas_training_image.image = tk_image

    def update_predicted_image(img_array):
        """
        Shows the predicted dataset image (28x28) on a dedicated canvas labeled "Predicted Image".
        """
        image = (img_array * 255).astype(np.uint8)
        pil_image = Image.fromarray(image, mode="L")
        pil_image = pil_image.resize((100, 100))
        tk_image = ImageTk.PhotoImage(pil_image)
        canvas_predicted_image.create_image(50, 50, image=tk_image, anchor=tk.CENTER)
        canvas_predicted_image.image = tk_image

    def update_architecture():
        """
        Visualizes the CNN architecture (input -> conv -> pool -> conv -> pool -> fc -> output).
        """
        if any(layer is None for layer in [conv1, conv2, pool1, pool2, fc]):
            return
        arch_info = build_architecture_list()
        draw_nn(canvas_nn, arch_info)

    # Create main Tkinter window
    root = tk.Tk()
    root.title("Convolutional Neural Network")
    root.state('zoomed')  # Start maximized
    root.protocol("WM_DELETE_WINDOW", on_close)

    # Frames for layout
    left_frame = ttk.Frame(root, width=250, relief=tk.GROOVE, padding=5)
    left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

    right_frame = ttk.Frame(root, relief=tk.GROOVE, padding=5)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    center_frame = ttk.Frame(root, relief=tk.GROOVE, padding=5)
    center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    # CNN architecture canvas
    canvas_nn = tk.Canvas(center_frame, width=600, height=800, bg="white")
    canvas_nn.pack(fill=tk.BOTH, expand=True)

    # Accuracy / Confusion frames
    top_right_frame = ttk.Frame(right_frame)
    top_right_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    bottom_right_frame = ttk.Frame(right_frame)
    bottom_right_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Accuracy Plot
    figure = Figure(figsize=(4, 2), dpi=100)
    acc_ax = figure.add_subplot(111)
    acc_ax.set_title("Training Accuracy")
    acc_ax.set_xlabel("Epochs")
    acc_canvas = FigureCanvasTkAgg(figure, master=top_right_frame)
    acc_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Confusion Matrix Plot
    confusion_fig = Figure(figsize=(4, 2), dpi=100)
    confusion_ax = confusion_fig.add_subplot(111)
    confusion_ax.set_title("Testing Confusion Matrix")
    confusion_canvas = FigureCanvasTkAgg(confusion_fig, master=bottom_right_frame)
    confusion_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # UI controls on the left_frame
    row_ = 0
    ttk.Label(left_frame, text="Epochs:").grid(row=row_, column=0, sticky=tk.W, pady=5)
    entry_epochs = ttk.Entry(left_frame, width=6)
    entry_epochs.insert(0, "10")  # default 10 epochs
    entry_epochs.grid(row=row_, column=1, pady=5)
    row_ += 1

    ttk.Label(left_frame, text="Learning Rate:").grid(row=row_, column=0, sticky=tk.W, pady=5)
    entry_lr = ttk.Entry(left_frame, width=6)
    entry_lr.insert(0, "0.01")
    entry_lr.grid(row=row_, column=1, pady=5)
    row_ += 1

    ttk.Label(left_frame, text="Conv1 Size:").grid(row=row_, column=0, sticky=tk.W, pady=3)
    entry_conv1_size = ttk.Entry(left_frame, width=6)
    entry_conv1_size.insert(0, "8")
    entry_conv1_size.grid(row=row_, column=1, pady=3)
    row_ += 1

    ttk.Label(left_frame, text="Pool1 Size:").grid(row=row_, column=0, sticky=tk.W, pady=3)
    entry_pool_size1 = ttk.Entry(left_frame, width=6)
    entry_pool_size1.insert(0, "2")
    entry_pool_size1.grid(row=row_, column=1, pady=3)
    row_ += 1

    ttk.Label(left_frame, text="Conv2 Size:").grid(row=row_, column=0, sticky=tk.W, pady=3)
    entry_conv2_size = ttk.Entry(left_frame, width=6)
    entry_conv2_size.insert(0, "16")
    entry_conv2_size.grid(row=row_, column=1, pady=3)
    row_ += 1

    ttk.Label(left_frame, text="Pool2 Size:").grid(row=row_, column=0, sticky=tk.W, pady=3)
    entry_pool_size2 = ttk.Entry(left_frame, width=6)
    entry_pool_size2.insert(0, "4")
    entry_pool_size2.grid(row=row_, column=1, pady=3)
    row_ += 1

    # Button to train the CNN
    ttk.Button(left_frame, text="Train CNN", command=train_cnn).grid(row=row_, columnspan=2, pady=15)
    row_ += 1

    training_sample_label = tk.StringVar(value="Sampled Data:")
    ttk.Label(left_frame, textvariable=training_sample_label).grid(row=row_, column=0,
                                                                   columnspan=2, sticky=tk.W)
    row_ += 1

    # Canvas to show a random training image from current batch
    canvas_training_image = tk.Canvas(left_frame, width=100, height=100, bg="white")
    canvas_training_image.grid(row=row_, columnspan=2, pady=15)
    row_ += 1

    # Progress bar for epochs
    progress_bar = ttk.Progressbar(left_frame, orient=tk.HORIZONTAL, length=150, mode='determinate')
    progress_bar.grid(row=row_, columnspan=2, pady=15)
    row_ += 1

    result_text = tk.StringVar()
    ttk.Label(left_frame, textvariable=result_text, font=("Arial", 12), wraplength=200)\
        .grid(row=row_, columnspan=2, pady=5)
    row_ += 1

    # Button to evaluate the CNN on the entire test set
    ttk.Button(left_frame, text="Test CNN", command=evaluate_on_test_set).grid(row=row_, columnspan=2, pady=15)
    row_ += 1

    # Button to pick a random dataset sample for prediction
    ttk.Button(left_frame, text="Predict From Dataset", command=predict_dataset_sample).grid(
        row=row_, columnspan=2, pady=15
    )
    row_ += 1

    ttk.Label(left_frame, text="Predicted Image:").grid(row=row_, column=0, columnspan=2, sticky=tk.W)
    row_ += 1

    # Canvas to show the randomly predicted image from dataset
    canvas_predicted_image = tk.Canvas(left_frame, width=100, height=100, bg="white")
    canvas_predicted_image.grid(row=row_, columnspan=2, pady=15)
    row_ += 1

    def initialize_layers():
        """
        Initializes the CNN layers based on UI inputs. This function
        reads the user-defined hyperparameters from the Tkinter entries.
        """
        lr_val = float(entry_lr.get())
        nf1 = int(entry_conv1_size.get())
        poolsize1 = int(entry_pool_size1.get())
        nf2 = int(entry_conv2_size.get())
        poolsize2 = int(entry_pool_size2.get())

        global conv1, pool1, conv2, pool2, fc

        # Convolution #1
        conv1 = Convolution(input_shape=(1, 28, 28), num_filters=nf1, filter_size=3,
                            stride=1, pad=0, lr=lr_val)
        pool1 = MaxPool(pool_size=poolsize1, stride=poolsize1)

        conv1_out_h = 28 - 3 + 1
        conv1_out_w = 28 - 3 + 1
        pool1_out_h = conv1_out_h // poolsize1
        pool1_out_w = conv1_out_w // poolsize1
        if poolsize1 > conv1_out_h or poolsize1 > conv1_out_w:
            raise ValueError(
                f"Pool1 size ({poolsize1}) is too large for Conv1 output dims "
                f"({conv1_out_h}, {conv1_out_w}).")

        # Convolution #2
        conv2_filter_size = 3
        conv2_stride = 1
        conv2_pad = 0

        # Attempt to ensure valid shape by decreasing filter size if needed
        while True:
            conv2_out_h = (pool1_out_h + 2*conv2_pad - conv2_filter_size) // conv2_stride + 1
            conv2_out_w = (pool1_out_w + 2*conv2_pad - conv2_filter_size) // conv2_stride + 1
            if conv2_out_h > 0 and conv2_out_w > 0:
                break
            conv2_filter_size -= 1
            if conv2_filter_size < 1:
                raise ValueError("Conv2 filter size/stride invalid. Cannot reduce further.")

        conv2 = Convolution(input_shape=(nf1, pool1_out_h, pool1_out_w),
                            num_filters=nf2, filter_size=conv2_filter_size,
                            stride=conv2_stride, pad=conv2_pad, lr=lr_val)

        pool2 = MaxPool(pool_size=poolsize2, stride=poolsize2)
        pool2_out_h = conv2_out_h // poolsize2
        pool2_out_w = conv2_out_w // poolsize2
        if poolsize2 > conv2_out_h or poolsize2 > conv2_out_w:
            raise ValueError(f"Pool2 size ({poolsize2}) is too large for Conv2 output dims "
                             f"({conv2_out_h}, {conv2_out_w}).")

        # Fully Connected layer
        fc = FullyConnected(input_size=nf2 * pool2_out_h * pool2_out_w, output_size=10, lr=lr_val)

        print(f"Conv2 filter size adjusted to {conv2_filter_size} to ensure valid dimensions.")
        print("Layers reinitialized.")
        print()
        return conv1, pool1, conv2, pool2, fc

    def build_architecture_list():
        """
        Creates a list describing each layer in the CNN
        (input -> conv -> pool -> conv -> pool -> fc -> output).
        We'll use it to draw a simple visualization.
        """
        arch_info = []
        arch_info.append(('input', min(10, 28 * 28)))
        arch_info.append(('conv', conv1.num_filters))
        arch_info.append(('pool', pool1.pool_size))
        arch_info.append(('conv', conv2.num_filters))
        arch_info.append(('pool', pool2.pool_size))
        fc_nodes = fc.weights.shape[0]
        arch_info.append(('fc', fc_nodes))
        arch_info.append(('output', 10))
        return arch_info

    root.mainloop()


# -----------------------------------
# draw_nn: Visualize the CNN model
# -----------------------------------
def draw_nn(canvas, arch_info):
    """
    Draws the CNN architecture on the canvas with color-coded shapes
    for each layer:
      input/output = circles,
      conv/fc = red rectangles,
      pool = green squares.
    """
    canvas.delete("all")
    width = canvas.winfo_width() or 600
    height = canvas.winfo_height() or 800

    x_spacing = width / (len(arch_info) + 1)
    max_nodes_to_draw = 16
    layer_positions = []

    for i, (layer_type, node_count) in enumerate(arch_info):
        x_center = (i + 1) * x_spacing
        shape_count = min(node_count, max_nodes_to_draw)

        y_spacing = height / (shape_count + 2)
        positions = []

        label_text = f"{layer_type.title()} ({node_count})"
        label_y = y_spacing * 0.5
        canvas.create_text(x_center, label_y, text=label_text, fill="black", font=("Arial", 10, "bold"))

        # Draw shapes for each neuron/filter
        for j in range(shape_count):
            y_center = (j + 1.5) * y_spacing
            if layer_type in ('input', 'output'):
                # Circle
                radius = 8
                canvas.create_oval(x_center - radius, y_center - radius,
                                   x_center + radius, y_center + radius,
                                   fill="darkblue", outline="black")
            elif layer_type == 'pool':
                # Green rectangle
                rect_w = 20
                rect_h = 20
                top_left_x = x_center - rect_w/2
                top_left_y = y_center - rect_h/2
                bottom_right_x = x_center + rect_w/2
                bottom_right_y = y_center + rect_h/2
                canvas.create_rectangle(top_left_x, top_left_y,
                                        bottom_right_x, bottom_right_y,
                                        fill="green", outline="black")
            elif layer_type in ('conv', 'fc'):
                # Red rectangle
                rect_w = 20
                rect_h = 20
                top_left_x = x_center - rect_w/2
                top_left_y = y_center - rect_h/2
                bottom_right_x = x_center + rect_w/2
                bottom_right_y = y_center + rect_h/2
                canvas.create_rectangle(top_left_x, top_left_y,
                                        bottom_right_x, bottom_right_y,
                                        fill="red", outline="black")
            positions.append((x_center, y_center))
        layer_positions.append(positions)

    # Connect the layers with lines
    for i in range(len(layer_positions) - 1):
        for pos1 in layer_positions[i]:
            for pos2 in layer_positions[i+1]:
                canvas.create_line(pos1[0], pos1[1], pos2[0], pos2[1], fill="black", width=1)


# --------------------------------------------
# main(): load MNIST data, shuffle, run UI
# --------------------------------------------
def main():
    global X_train, y_train_one_hot, X_test, y_test

    # Load data from 'mnist-original.mat' file
    mat_data = loadmat('mnist-original.mat')
    X_all = mat_data['data']
    y_all = mat_data['label']
    X_all = X_all.reshape((28, 28, -1))
    y_all = y_all.flatten().astype(int)

    # Shuffle entire dataset to mix all classes randomly
    indices = np.arange(X_all.shape[2])
    np.random.shuffle(indices)
    X_all = X_all[:, :, indices]
    y_all = y_all[indices]

    # Split shuffled data into train (60k) and test (10k)
    X_train = X_all[:, :, :60000]
    y_train = y_all[:60000]
    X_test = X_all[:, :, 60000:]
    y_test_ = y_all[60000:]

    # Convert labels to one-hot vectors
    y_train_one_hot = one_hot_encode(y_train, 10)
    y_test = one_hot_encode(y_test_, 10)

    # Reshape to (N, 1, 28, 28), converting to float
    X_train = np.transpose(X_train, (2, 0, 1))[:, np.newaxis, :, :].astype(np.float32) / 255.0
    X_test  = np.transpose(X_test,  (2, 0, 1))[:, np.newaxis, :, :].astype(np.float32) / 255.0

    # Normalize data by mean/std of training set
    mean = np.mean(X_train, axis=(0,2,3), keepdims=True)
    std  = np.std(X_train,  axis=(0,2,3), keepdims=True)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test  = (X_test  - mean) / (std + 1e-8)

    print("MNIST data loaded and shuffled successfully.")
    print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape,   "y_test shape:",  y_test.shape)
    print()

    # Create the GUI
    create_ui()


# Entry point
if __name__ == "__main__":
    main()
