import numpy as np


def compute_accuracy(X, y, net, batch_size=100):
    N = X.shape[0]
    correct = 0
    for start_idx in range(0, N, batch_size):
        end_idx = start_idx + batch_size
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]

        out = net.forward(X_batch)
        preds = np.argmax(out, axis=1)
        correct += np.sum(preds == y_batch)

    return correct / N


def train_network(
    net,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=5,
    batch_size=64,
    learning_rate=0.01,
    print_every=100,
):
    num_batches = X_train.shape[0] // batch_size

    for epoch in range(1, epochs + 1):
        print(f"--- Epoch {epoch}/{epochs} ---")

        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        epoch_loss = 0.0
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            loss = net.loss_and_backward(X_batch, y_batch)
            epoch_loss += loss

            # Parameter update (SGD)
            for layer in net.layers:
                if hasattr(layer, "W"):
                    layer.W -= learning_rate * layer.dW
                if hasattr(layer, "b"):
                    layer.b -= learning_rate * layer.db

            # Print progress
            if (i + 1) % print_every == 0:
                print(f"Epoch {epoch}, Batch {i+1}/{num_batches}, Loss: {loss:.4f}")

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")

        test_acc = compute_accuracy(X_test, y_test, net)
        print(f"Test Accuracy after Epoch {epoch}: {test_acc * 100:.2f}%\n")

    print("Training completed.")
