import numpy as np


def one_hot(labels, num_classes=10):
    # Convert integer labels to one-hot vectors
    N = labels.shape[0]
    one_hot_labels = np.zeros((N, num_classes))
    one_hot_labels[np.arange(N), labels] = 1.0
    return one_hot_labels


def im2col(X, kernel_size, stride, padding):
    N, C, H, W = X.shape
    KH, KW = kernel_size, kernel_size
    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W + 2 * padding - KW) // stride + 1

    # Pad input
    X_padded = np.pad(
        X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant"
    )
    col = np.zeros((N, C, KH, KW, H_out, W_out))

    for y in range(KH):
        y_max = y + stride * H_out
        for x in range(KW):
            x_max = x + stride * W_out
            col[:, :, y, x, :, :] = X_padded[:, :, y:y_max:stride, x:x_max:stride]

    # (N, C, KH, KW, H_out, W_out) -> (N * H_out * W_out, C * KH * KW)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * H_out * W_out, -1)
    return col


def col2im(col, X_shape, kernel_size, stride, padding):
    N, C, H, W = X_shape
    KH, KW = kernel_size, kernel_size
    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W + 2 * padding - KW) // stride + 1

    col = col.reshape(N, H_out, W_out, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)
    X_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding))

    for y in range(KH):
        y_max = y + stride * H_out
        for x in range(KW):
            x_max = x + stride * W_out
            X_padded[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    if padding > 0:
        return X_padded[:, :, padding:-padding, padding:-padding]
    return X_padded


class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        # Xavier initialization for filters
        limit = np.sqrt(6.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.uniform(
            -limit, limit, (out_channels, in_channels, kernel_size, kernel_size)
        )
        self.b = np.zeros((out_channels,))
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.out_height = None
        self.out_width = None

    def forward(self, X):
        self.X = X
        N, C, H, W = X.shape
        out_channels, _, KH, KW = self.W.shape

        self.col = im2col(X, KH, self.stride, self.padding)
        W_reshaped = self.W.reshape(out_channels, -1)

        out = self.col.dot(W_reshaped.T) + self.b
        H_out = (H + 2 * self.padding - KH) // self.stride + 1
        W_out = (W + 2 * self.padding - KW) // self.stride + 1
        out = out.reshape(N, H_out, W_out, out_channels).transpose(0, 3, 1, 2)

        self.W_reshaped = W_reshaped
        self.out_height = H_out
        self.out_width = W_out
        return out

    def backward(self, dOut):
        N, C_out, H_out, W_out = dOut.shape
        out_channels, in_channels, KH, KW = self.W.shape

        dOut_reshaped = dOut.transpose(0, 2, 3, 1).reshape(-1, C_out)
        self.db = np.sum(dOut_reshaped, axis=0)
        self.dW = dOut_reshaped.T.dot(self.col).reshape(self.W.shape)

        dCol = dOut_reshaped.dot(self.W_reshaped)
        dX = col2im(dCol, self.X.shape, KH, self.stride, self.padding)

        return dX


class ReLULayer:
    def forward(self, X):
        self.X = X
        self.out_shape = X.shape  # Store shape for printing
        return np.maximum(0, X)

    def backward(self, dOut):
        dX = dOut * (self.X > 0)
        return dX


class MaxPoolLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, X):
        self.X = X
        N, C, H, W = X.shape
        HH = H // self.pool_size
        WW = W // self.pool_size

        out = np.zeros((N, C, HH, WW))
        self.max_indices = np.zeros((N, C, HH, WW, 2), dtype=int)

        for n in range(N):
            for c in range(C):
                for i in range(HH):
                    for j in range(WW):
                        region = X[
                            n,
                            c,
                            i * self.stride : i * self.stride + self.pool_size,
                            j * self.stride : j * self.stride + self.pool_size,
                        ]
                        max_val = np.max(region)
                        out[n, c, i, j] = max_val
                        idx = np.unravel_index(np.argmax(region), region.shape)
                        self.max_indices[n, c, i, j] = [
                            idx[0] + i * self.stride,
                            idx[1] + j * self.stride,
                        ]

        self.out_channels = C
        self.out_height = HH
        self.out_width = WW
        return out

    def backward(self, dOut):
        N, C, HH, WW = dOut.shape
        dX = np.zeros_like(self.X)

        for n in range(N):
            for c in range(C):
                for i in range(HH):
                    for j in range(WW):
                        (max_i, max_j) = self.max_indices[n, c, i, j]
                        dX[n, c, max_i, max_j] += dOut[n, c, i, j]

        return dX


class FlattenLayer:
    def forward(self, X):
        self.X_shape = X.shape
        N, C, H, W = X.shape
        self.out_dim = C * H * W
        return X.reshape(N, -1)

    def backward(self, dOut):
        return dOut.reshape(self.X_shape)


class FullyConnectedLayer:
    def __init__(self, in_dim, out_dim):
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.W = np.random.uniform(-limit, limit, (in_dim, out_dim))
        self.b = np.zeros((out_dim,))
        self.out_dim = out_dim

    def forward(self, X):
        self.X = X
        return X.dot(self.W) + self.b

    def backward(self, dOut):
        dX = dOut.dot(self.W.T)
        self.dW = self.X.T.dot(dOut)
        self.db = np.sum(dOut, axis=0)
        return dX


class SoftmaxLayer:
    def forward(self, X):
        shift_X = X - np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(shift_X)
        self.out = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        self.out_dim = X.shape[1]
        return self.out

    def backward(self, dOut):
        return dOut


class Network:
    def __init__(self):
        # A simple CNN for MNIST
        self.layers = [
            ConvLayer(
                in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1
            ),
            ReLULayer(),
            MaxPoolLayer(pool_size=2, stride=2),
            FlattenLayer(),
            FullyConnectedLayer(in_dim=8 * 14 * 14, out_dim=10),
            SoftmaxLayer(),
        ]

    def forward(self, X):
        # Store input shape the first time forward is called
        if not hasattr(self, "input_shape"):
            self.input_shape = X.shape  # (N, C, H, W)
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def compute_loss(self, out, y):
        N = y.shape[0]
        y_one_hot_vec = one_hot(y, 10)
        log_out = np.log(out + 1e-9)
        loss = -np.sum(y_one_hot_vec * log_out) / N
        return loss, y_one_hot_vec

    def loss_and_backward(self, X, y):
        out = self.forward(X)
        loss, y_one_hot_vec = self.compute_loss(out, y)
        N = y.shape[0]
        dOut = (out - y_one_hot_vec) / N

        for layer in reversed(self.layers):
            dOut = layer.backward(dOut)

        return loss

    def forward_with_intermediates(self, X):
        """
        Runs a forward pass through the network but returns
        a dictionary of intermediate outputs after each layer.
        """
        intermediates = {}
        out = X

        # input is just X
        intermediates["input"] = out

        # Layer order: Conv -> ReLU -> Pool -> Flatten -> FC -> Softmax
        # We know net.layers = [ConvLayer, ReLULayer, MaxPoolLayer, FlattenLayer, FullyConnectedLayer, SoftmaxLayer]

        # 0: Conv
        out = self.layers[0].forward(out)  # Conv
        intermediates["conv"] = out

        # 1: ReLU
        out = self.layers[1].forward(out)  # ReLU
        intermediates["relu"] = out

        # 2: MaxPool
        out = self.layers[2].forward(out)  # Pool
        intermediates["pool"] = out

        # 3: Flatten
        out = self.layers[3].forward(out)  # Flatten
        intermediates["flatten"] = out

        # 4: FullyConnected
        out = self.layers[4].forward(out)  # FC
        intermediates["fc"] = out

        # 5: Softmax
        out = self.layers[5].forward(out)  # Softmax
        intermediates["softmax"] = out

        return intermediates

    def print_structure(self):
        print("Network Structure:")
        # Print input dimensions
        if hasattr(self, "input_shape"):
            N, C, H, W = self.input_shape
            print(f"Input: {C}x{H}x{W}")

        for i, layer in enumerate(self.layers):
            layer_type = type(layer).__name__
            if isinstance(layer, ConvLayer):
                print(
                    f"Layer {i}: {layer_type} | Output: {layer.out_channels}x{layer.out_height}x{layer.out_width}"
                )
            elif isinstance(layer, MaxPoolLayer):
                print(
                    f"Layer {i}: {layer_type} | Output: {layer.out_channels}x{layer.out_height}x{layer.out_width}"
                )
            elif isinstance(layer, FlattenLayer):
                print(f"Layer {i}: {layer_type} | Output: {layer.out_dim} neurons")
            elif isinstance(layer, FullyConnectedLayer):
                print(f"Layer {i}: {layer_type} | Output: {layer.out_dim} neurons")
            elif isinstance(layer, SoftmaxLayer):
                print(f"Layer {i}: {layer_type} | Output: {layer.out_dim} classes")
            elif isinstance(layer, ReLULayer):
                shape = getattr(layer, "out_shape", None)
                if shape is not None:
                    if len(shape) == 4:
                        _, C_, H_, W_ = shape
                        print(f"Layer {i}: {layer_type} | Output: {C_}x{H_}x{W_}")
                    else:
                        # assume flatten shape (N, dim)
                        _, dim = shape
                        print(f"Layer {i}: {layer_type} | Output: {dim} neurons")
                else:
                    print(f"Layer {i}: {layer_type} | Shape not recorded.")
            else:
                print(f"Layer {i}: {layer_type} | Shape not known.")


if __name__ == "__main__":
    # Quick test
    X_dummy = np.random.randn(2, 1, 28, 28).astype(np.float32)
    y_dummy = np.array([3, 1])
    net = Network()
    out = net.forward(X_dummy)
    net.print_structure()
    loss = net.loss_and_backward(X_dummy, y_dummy)
    print("Loss after backward:", loss)
