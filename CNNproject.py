import numpy as np
import tkinter as tk
from tkinter import Canvas
import gzip
import struct


class ConvBlock:
    """
    A simple 2D convolution layer operating on single-channel images.
    """

    def __init__(self, num_filters, kernel_size, input_dims):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_dims = input_dims
        # Initialize weights and biases
        self.kernels = np.random.randn(num_filters, kernel_size, kernel_size) * 0.1
        self.biases = np.zeros((num_filters,))

    def forward(self, inp_image):
        self.inp_image = inp_image
        out_h = self.input_dims[0] - self.kernel_size + 1
        out_w = self.input_dims[1] - self.kernel_size + 1
        self.output = np.zeros((out_h, out_w, self.num_filters))

        for f in range(self.num_filters):
            for i in range(out_h):
                for j in range(out_w):
                    region = inp_image[i:i + self.kernel_size, j:j + self.kernel_size]
                    self.output[i, j, f] = np.sum(region * self.kernels[f]) + self.biases[f]
        return self.output

    def backward(self, grad_out, lr):
        grad_k = np.zeros_like(self.kernels)
        grad_b = np.zeros_like(self.biases)
        grad_in = np.zeros_like(self.inp_image)

        out_h, out_w, _ = grad_out.shape

        for f in range(self.num_filters):
            for i in range(out_h):
                for j in range(out_w):
                    patch = self.inp_image[i:i + self.kernel_size, j:j + self.kernel_size]
                    grad_k[f] += grad_out[i, j, f] * patch
                    grad_in[i:i + self.kernel_size, j:j + self.kernel_size] += grad_out[i, j, f] * self.kernels[f]
                    grad_b[f] += grad_out[i, j, f]

        self.kernels -= lr * grad_k
        self.biases -= lr * grad_b
        return grad_in


class MaxPool2x2:
    """
    A max-pooling layer with a 2x2 filter.
    """

    def __init__(self):
        self.pool_size = 2

    def forward(self, inp_tensor):
        self.inp_tensor = inp_tensor
        h, w, c = inp_tensor.shape
        out_h = h // self.pool_size
        out_w = w // self.pool_size
        self.output = np.zeros((out_h, out_w, c))

        for ch in range(c):
            for i in range(0, h, self.pool_size):
                for j in range(0, w, self.pool_size):
                    block = inp_tensor[i:i + self.pool_size, j:j + self.pool_size, ch]
                    self.output[i // self.pool_size, j // self.pool_size, ch] = np.max(block)
        return self.output

    def backward(self, grad_out):
        grad_in = np.zeros_like(self.inp_tensor)
        out_h, out_w, c = grad_out.shape

        for ch in range(c):
            for i in range(out_h):
                for j in range(out_w):
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size
                    window = self.inp_tensor[start_i:start_i + self.pool_size, start_j:start_j + self.pool_size, ch]
                    max_val = np.max(window)
                    mask = (window == max_val)
                    grad_in[start_i:start_i + self.pool_size, start_j:start_j + self.pool_size, ch] += grad_out[
                                                                                                           i, j, ch] * mask
        return grad_in


class DenseLayer:
    """
    A fully connected layer.
    """

    def __init__(self, in_features, out_features):
        self.weights = np.random.randn(in_features, out_features) * 0.1
        self.biases = np.zeros((out_features,))

    def forward(self, x):
        self.input_ = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, grad_out, lr):
        grad_w = np.dot(self.input_.T, grad_out)
        grad_b = np.sum(grad_out, axis=0)
        grad_in = np.dot(grad_out, self.weights.T)

        self.weights -= lr * grad_w
        self.biases -= lr * grad_b
        return grad_in


def apply_relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(float)


def apply_softmax(x):
    shift_x = x - np.max(x, axis=1, keepdims=True)
    exps = np.exp(shift_x)
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))


def calc_accuracy(preds, labels):
    pred_labels = np.argmax(preds, axis=1)
    true_labels = np.argmax(labels, axis=1)
    return np.mean(pred_labels == true_labels)


class SimpleCNN:
    """
    Encapsulates the CNN layers and forward/backward logic.
    Architecture: Conv -> ReLU -> Pool -> Flatten -> Dense -> ReLU -> Dense -> Softmax
    """

    def __init__(self):
        self.conv = ConvBlock(num_filters=8, kernel_size=3, input_dims=(28, 28))
        self.pool = MaxPool2x2()
        # After conv(3x3) + pool(2x2): 28-3+1=26, then 26/2=13
        # Output shape after pool: 13x13x8 = 1352
        self.fc1 = DenseLayer(in_features=1352, out_features=64)
        self.fc2 = DenseLayer(in_features=64, out_features=10)

    def forward(self, x_batch):
        # x_batch shape: (batch, 28,28)
        batch_size = x_batch.shape[0]
        self.c_out_list = []
        self.a_out_list = []
        self.p_out_list = []

        for i in range(batch_size):
            c_out = self.conv.forward(x_batch[i])
            a_out = apply_relu(c_out)
            p_out = self.pool.forward(a_out)
            self.c_out_list.append(c_out)
            self.a_out_list.append(a_out)
            self.p_out_list.append(p_out)

        p_arr = np.array(self.p_out_list).reshape(batch_size, -1)
        self.fc1_out = apply_relu(self.fc1.forward(p_arr))
        self.logits = self.fc2.forward(self.fc1_out)
        self.probs = apply_softmax(self.logits)
        return self.probs

    def backward(self, y_true, lr):
        # Backprop starts here
        grad_out = self.probs - y_true  # dLoss/dLogits before softmax (cross-entropy)
        d_fc2 = self.fc2.backward(grad_out, lr)
        d_fc1 = self.fc1.backward(d_fc2 * relu_grad(self.fc1_out), lr)

        # Reshape to match pooled output
        batch_size = y_true.shape[0]
        d_pool_out = d_fc1.reshape(batch_size, 13, 13, 8)

        for i in range(batch_size):
            d_p = self.pool.backward(d_pool_out[i])
            d_r = d_p * relu_grad(self.a_out_list[i])
            self.conv.backward(d_r, lr)


def load_mnist():
    paths = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }

    def read_gz(path):
        with gzip.open(path, 'rb') as f:
            return f.read()

    def parse_imgs(data):
        _, num, rows, cols = struct.unpack(">IIII", data[:16])
        return np.frombuffer(data[16:], dtype=np.uint8).reshape(num, rows, cols)

    def parse_lbls(data):
        _, num = struct.unpack(">II", data[:8])
        return np.frombuffer(data[8:], dtype=np.uint8)

    tr_imgs = parse_imgs(read_gz(paths["train_images"]))
    tr_lbls = parse_lbls(read_gz(paths["train_labels"]))
    ts_imgs = parse_imgs(read_gz(paths["test_images"]))
    ts_lbls = parse_lbls(read_gz(paths["test_labels"]))
    return tr_imgs, tr_lbls, ts_imgs, ts_lbls


def show_sample_digit(imgs, lbls):
    """Display a random digit from the dataset."""
    root = tk.Tk()
    root.title("Sample MNIST Digit")

    canvas = Canvas(root, width=280, height=280)
    canvas.pack()

    idx = np.random.randint(len(imgs))
    digit = imgs[idx]
    actual_lbl = lbls[idx]

    for i in range(28):
        for j in range(28):
            val = 255 - digit[i, j]
            color_hex = f"#{val:02x}{val:02x}{val:02x}"
            canvas.create_rectangle(j * 10, i * 10, (j + 1) * 10, (i + 1) * 10, fill=color_hex, outline=color_hex)

    label_info = tk.Label(root, text=f"Label: {actual_lbl}", font=("Arial", 16))
    label_info.pack()
    root.mainloop()


def one_hot_encode(labels, n_class=10):
    arr = np.zeros((labels.size, n_class))
    arr[np.arange(labels.size), labels] = 1
    return arr


def normalize_pixels(images):
    return images / 255.0


def create_batches(X, Y, batch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start in range(0, X.shape[0], batch_size):
        end = start + batch_size
        batch_indices = indices[start:end]
        yield X[batch_indices], Y[batch_indices]


def train_cnn(model, X_train, Y_train, epochs, batch_size, lr):
    print("Initiating training procedure...")
    for epoch in range(1, epochs + 1):
        print(f"Epoch [{epoch}/{epochs}]")
        total_loss = 0.0
        total_acc = 0.0
        count_batches = 0

        for batch_i, (Xb, Yb) in enumerate(create_batches(X_train, Y_train, batch_size), start=1):
            # Forward
            preds = model.forward(Xb)
            loss_val = cross_entropy_loss(preds, Yb)
            acc_val = calc_accuracy(preds, Yb)

            total_loss += loss_val
            total_acc += acc_val
            count_batches += 1

            # Backward
            model.backward(Yb, lr)

            if batch_i % 100 == 0:
                print(
                    f"  Processed {batch_i * batch_size} samples | Batch Loss: {loss_val:.4f} | Batch Acc: {acc_val * 100:.1f}%")

        avg_loss = total_loss / count_batches
        avg_acc = total_acc / count_batches
        print(f"Completed Epoch {epoch} - Avg Loss: {avg_loss:.4f}, Train Acc: {avg_acc * 100:.2f}%\n")


def evaluate_cnn(model, X_test, Y_test, batch_size=128):
    probs_list = []
    for start in range(0, X_test.shape[0], batch_size):
        end = start + batch_size
        X_segment = X_test[start:end]
        out_probs = model.forward(X_segment)
        probs_list.append(out_probs)

    all_probs = np.vstack(probs_list)
    test_acc = calc_accuracy(all_probs, Y_test)
    return test_acc


def main():
    train_imgs, train_lbls, test_imgs, test_lbls = load_mnist()

    # Show a digit before training
    show_sample_digit(train_imgs, train_lbls)

    # Preprocessing
    X_train = normalize_pixels(train_imgs)
    X_test = normalize_pixels(test_imgs)
    Y_train = one_hot_encode(train_lbls)
    Y_test = one_hot_encode(test_lbls)

    # Instantiate model
    model = SimpleCNN()

    # Hyperparameters
    learn_rate = 0.01
    n_epochs = 1
    b_size = 16

    # Train the model
    train_cnn(model, X_train, Y_train, n_epochs, b_size, learn_rate)

    # Evaluate on test data
    test_accuracy = evaluate_cnn(model, X_test, Y_test, 128)
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
