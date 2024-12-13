import numpy as np
import tkinter as tk
from tkinter import Canvas
import gzip
import struct


# -------------------------------
# Layers and Utility Functions
# -------------------------------
class Convolution2DLayer:
    def __init__(self, filter_count, kernel_size, input_dim):
        self.num_filters = filter_count
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.kernels = np.random.randn(filter_count, kernel_size, kernel_size) * 0.1
        self.biases = np.zeros((filter_count,))

    def forward(self, input_image):
        self.input_image = input_image
        out_height = self.input_dim[0] - self.kernel_size + 1
        out_width = self.input_dim[1] - self.kernel_size + 1
        self.output = np.zeros((out_height, out_width, self.num_filters))

        for f in range(self.num_filters):
            for i in range(out_height):
                for j in range(out_width):
                    patch = input_image[i:i + self.kernel_size, j:j + self.kernel_size]
                    self.output[i, j, f] = np.sum(patch * self.kernels[f]) + self.biases[f]
        return self.output

    def backward(self, d_out, lr):
        dk = np.zeros_like(self.kernels)
        db = np.zeros_like(self.biases)
        d_input = np.zeros_like(self.input_image)

        out_height, out_width, _ = d_out.shape

        for f in range(self.num_filters):
            for i in range(out_height):
                for j in range(out_width):
                    region = self.input_image[i:i + self.kernel_size, j:j + self.kernel_size]
                    dk[f] += d_out[i, j, f] * region
                    d_input[i:i + self.kernel_size, j:j + self.kernel_size] += d_out[i, j, f] * self.kernels[f]
                    db[f] += d_out[i, j, f]

        self.kernels -= lr * dk
        self.biases -= lr * db

        return d_input


class MaxPoolingLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        h, w, c = input_tensor.shape
        out_height = h // self.pool_size
        out_width = w // self.pool_size
        self.output = np.zeros((out_height, out_width, c))

        for f in range(c):
            for i in range(0, h, self.pool_size):
                for j in range(0, w, self.pool_size):
                    region = input_tensor[i:i + self.pool_size, j:j + self.pool_size, f]
                    self.output[i // self.pool_size, j // self.pool_size, f] = np.max(region)

        return self.output

    def backward(self, d_out):
        d_input = np.zeros_like(self.input_tensor)
        out_height, out_width, c = d_out.shape

        for f in range(c):
            for i in range(out_height):
                for j in range(out_width):
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size
                    region = self.input_tensor[start_i:start_i + self.pool_size, start_j:start_j + self.pool_size, f]
                    max_val = np.max(region)
                    mask = (region == max_val)
                    d_input[start_i:start_i + self.pool_size, start_j:start_j + self.pool_size, f] += d_out[
                                                                                                          i, j, f] * mask

        return d_input


class DenseLayer:
    def __init__(self, in_size, out_size):
        self.weights = np.random.randn(in_size, out_size) * 0.1
        self.biases = np.zeros((out_size,))

    def forward(self, inp):
        self.input_data = inp
        return np.dot(inp, self.weights) + self.biases

    def backward(self, d_out, lr):
        dw = np.dot(self.input_data.T, d_out)
        db = np.sum(d_out, axis=0)
        d_input = np.dot(d_out, self.weights.T)

        self.weights -= lr * dw
        self.biases -= lr * db
        return d_input


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    exp_shifted = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


def categorical_crossentropy(y_pred, y_true):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))


def accuracy(predictions, labels):
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(labels, axis=1)
    return np.mean(pred_labels == true_labels)


# -------------------------------
# MNIST Data Loading
# -------------------------------
def load_mnist_data():
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    def read_gz(file_path):
        with gzip.open(file_path, 'rb') as f:
            return f.read()

    def parse_images(data):
        magic, num, rows, cols = struct.unpack(">IIII", data[:16])
        return np.frombuffer(data[16:], dtype=np.uint8).reshape(num, rows, cols)

    def parse_labels(data):
        magic, num = struct.unpack(">II", data[:8])
        return np.frombuffer(data[8:], dtype=np.uint8)

    train_images = parse_images(read_gz(files["train_images"]))
    train_labels = parse_labels(read_gz(files["train_labels"]))
    test_images = parse_images(read_gz(files["test_images"]))
    test_labels = parse_labels(read_gz(files["test_labels"]))

    return train_images, train_labels, test_images, test_labels


def visualize_digit_sample(images, labels):
    root = tk.Tk()
    root.title("MNIST Sample Visualization")

    canvas = Canvas(root, width=280, height=280)
    canvas.pack()

    idx = np.random.randint(len(images))
    digit_img = images[idx]
    digit_label = labels[idx]

    for i in range(28):
        for j in range(28):
            val = 255 - digit_img[i, j]
            color_code = f"#{val:02x}{val:02x}{val:02x}"
            canvas.create_rectangle(j * 10, i * 10, (j + 1) * 10, (i + 1) * 10, fill=color_code, outline=color_code)

    lbl = tk.Label(root, text=f"Label: {digit_label}", font=("Arial", 16))
    lbl.pack()

    root.mainloop()


def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


def preprocess_data(images):
    return images / 255.0


def create_mini_batches(X, y, batch_size):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    for start in range(0, X.shape[0], batch_size):
        end = start + batch_size
        batch_idx = idx[start:end]
        yield X[batch_idx], y[batch_idx]


if __name__ == "__main__":
    # Load the data
    train_imgs, train_lbls, test_imgs, test_lbls = load_mnist_data()

    # Visualize one random digit (optional)
    visualize_digit_sample(train_imgs, train_lbls)

    # Preprocess data
    X_train = preprocess_data(train_imgs)
    X_test = preprocess_data(test_imgs)

    # One-hot encoding
    y_train = one_hot_encode(train_lbls, 10)
    y_test = one_hot_encode(test_lbls, 10)

    # Hyperparameters
    learning_rate = 0.01
    epochs = 1
    batch_size = 16

    # Model Architecture
    conv = Convolution2DLayer(filter_count=8, kernel_size=3, input_dim=(28, 28))
    pool = MaxPoolingLayer(pool_size=2)
    fc1 = DenseLayer(in_size=1352, out_size=64)
    fc2 = DenseLayer(in_size=64, out_size=10)

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        for batch_num, (X_batch, y_batch) in enumerate(create_mini_batches(X_train, y_train, batch_size), start=1):
            batch_size_curr = X_batch.shape[0]

            # Store outputs for each layer
            c_out_list = []
            a_out_list = []
            p_out_list = []

            # Forward pass for each image in the batch
            for img_i in range(batch_size_curr):
                c_out = conv.forward(X_batch[img_i])  # shape (26,26,8)
                a_out = relu(c_out)  # shape (26,26,8)
                p_out = pool.forward(a_out)  # shape (13,13,8)

                c_out_list.append(c_out)
                a_out_list.append(a_out)
                p_out_list.append(p_out)

            p_out_batch = np.array(p_out_list)  # (batch,13,13,8)
            flat = p_out_batch.reshape(batch_size_curr, -1)
            fc1_out = relu(fc1.forward(flat))
            logits = fc2.forward(fc1_out)
            probs = softmax(logits)

            loss = categorical_crossentropy(probs, y_batch)
            acc = accuracy(probs, y_batch)

            epoch_loss += loss
            epoch_acc += acc
            num_batches += 1

            # Backprop
            d_loss = probs - y_batch
            d_fc2 = fc2.backward(d_loss, learning_rate)
            d_fc1 = fc1.backward(d_fc2 * relu_derivative(fc1_out), learning_rate)

            # d_fc1 shape = (batch,1352), reshape to (batch,13,13,8)
            d_p_out = d_fc1.reshape(batch_size_curr, 13, 13, 8)

            # Backprop through pooling and conv layers for each image
            for img_i in range(batch_size_curr):
                # d_p_out[img_i] corresponds to gradient after pooling layer
                d_p = pool.backward(d_p_out[img_i])  # shape (26,26,8)
                # Apply ReLU derivative on a_out_list (not p_out_list!)
                d_r = d_p * relu_derivative(a_out_list[img_i])
                conv.backward(d_r, learning_rate)

            if batch_num % 100 == 0:
                print(
                    f"  Processed {batch_num * batch_size} samples - Current Batch Loss: {loss:.4f}, Acc: {acc * 100:.2f}%")

        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        print(f"End of Epoch {epoch} - Loss: {avg_loss:.4f}, Training Accuracy: {avg_acc * 100:.2f}%\n")

    # Testing
    print("Testing the model...")
    test_probs_list = []
    test_batch_size = 128
    for start in range(0, X_test.shape[0], test_batch_size):
        end = start + test_batch_size
        X_test_batch = X_test[start:end]
        c_out_test_list = []
        a_out_test_list = []
        p_out_test_list = []

        for img_i in range(X_test_batch.shape[0]):
            c_out_test = conv.forward(X_test_batch[img_i])
            a_out_test = relu(c_out_test)
            p_out_test = pool.forward(a_out_test)

            p_out_test_list.append(p_out_test)

        test_p_out_batch = np.array(p_out_test_list)
        test_flat = test_p_out_batch.reshape(X_test_batch.shape[0], -1)
        fc1_out_test = relu(fc1.forward(test_flat))
        logits_test = fc2.forward(fc1_out_test)
        probs_test = softmax(logits_test)
        test_probs_list.append(probs_test)

    test_probs_all = np.vstack(test_probs_list)
    test_accuracy = accuracy(test_probs_all, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")