import os
import struct
import numpy as np


def load_images(filename):
    with open(filename, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("Invalid magic number for images file!")
        data = np.fromfile(f, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols)
        return data


def load_labels(filename):
    with open(filename, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Invalid magic number for labels file!")
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def load_mnist_data(data_dir="data"):
    train_images_path = os.path.join(data_dir, "train-images.idx3-ubyte")
    train_labels_path = os.path.join(data_dir, "train-labels.idx1-ubyte")
    test_images_path = os.path.join(data_dir, "t10k-images.idx3-ubyte")
    test_labels_path = os.path.join(data_dir, "t10k-labels.idx1-ubyte")

    X_train = load_images(train_images_path)
    y_train = load_labels(train_labels_path)
    X_test = load_images(test_images_path)
    y_test = load_labels(test_labels_path)

    # Normalize the image data to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # For a CNN, reshape to (N, 1, 28, 28)
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)

    return X_train, y_train, X_test, y_test
