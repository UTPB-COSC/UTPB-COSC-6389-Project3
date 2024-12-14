from src.data import load_mnist_data
from src.network import Network
from src.train import train_network
import numpy as np


def main():
    # Load the MNIST data
    X_train, y_train, X_test, y_test = load_mnist_data()

    # Print the shapes to verify
    print("Training images shape:", X_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Test images shape:", X_test.shape)
    print("Test labels shape:", y_test.shape)

    # Initialize the network
    net = Network()

    # Train the network with vectorized convolution
    train_network(
        net,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=5,
        batch_size=64,
        learning_rate=0.01,
        print_every=1,
    )


if __name__ == "__main__":
    main()
