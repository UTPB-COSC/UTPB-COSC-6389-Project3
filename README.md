# UTPB-COSC-6389-Project3
# CNN Training Visualization with Custom Implementation

This project implements a Convolutional Neural Network (CNN) from scratch and provides a user-friendly graphical interface to visualize the training process and architecture. The CNN is designed for binary image classification tasks, such as distinguishing between cats and dogs.

## Features

- **Custom CNN Implementation**: A convolutional neural network built entirely from scratch without using machine learning libraries like TensorFlow or PyTorch.
- **Training Visualization**: Real-time plots of training loss and accuracy displayed during the training process.
- **CNN Architecture Visualization**: A graphical representation of the CNN architecture, including layers and their shapes.
- **Dynamic Updates**: Training progress, including epoch number, loss, and accuracy, dynamically updated in the GUI.
- **Image Loader**: Load and preprocess images for training, with support for binary classification (e.g., cats vs. dogs).

## CNN Architecture

The CNN implemented in this project consists of the following layers:

1. **Input Layer**: Accepts images of size `(128, 128, 3)`.
2. **Convolutional Layer (Conv2D)**: Applies 8 filters of size `(3, 3)` with ReLU activation, resulting in an output shape of `(126, 126, 8)`.
3. **Max Pooling Layer**: Reduces the spatial dimensions by applying a pool size of `(2, 2)`, resulting in an output shape of `(63, 63, 8)`.
4. **Fully Connected Layer**: Flattens the output to a vector of size `30752`, followed by a dense layer with 2 neurons for binary classification.
5. **Output Layer**: Outputs probabilities for the two classes using the Softmax activation function.

**CNN Architecture Visualization**
   - The architecture of the CNN is dynamically displayed on the right side of the UI. 
   - Each layer is represented as a rectangular box, with the following information displayed inside:
     - **Layer Type**: Shows the type of the layer (e.g., Input, Convolutional, ReLU, etc.).
     - **Shape/Output Shape**: Displays the dimensions of the layer's output (e.g., `(128, 128, 3)` for the input layer or `(126, 126, 8)` for the convolutional layer).
   - The UI also dynamically updates the layer colors based on the training progress. 
     - **Green**: Indicates a layer with updated weights.
     - **Red**: Indicates a layer with inactive or untrained weights.
   - Optionally, activation outputs can be visualized under each layer, showing the number of activations in each layer during training.


## GUI Overview

### Layout
- **Training Plot**: A real-time plot of the training loss, located on the left side of the window.
- **Information Panel**: Displays the current epoch, loss, and accuracy during training, located on the right side.
- **CNN Architecture Visualization**: A visual representation of the CNN layers and shapes, also on the right side.

### Interactions
- **Start Training**: A button to initiate the training process.
- **Dynamic Updates**: The GUI updates in real-time as the training progresses, showing changes in loss, accuracy, and epoch count.

![img.png](img.png)

## How to Run

1. **Prerequisites**:
   - Python 3.8 or higher
   - Required libraries: `numpy`, `tkinter`, `Pillow`, `matplotlib`

   Install dependencies using:
   ```bash
   pip install numpy pillow matplotlib

2. **UI Operation**:
  - Click the "Start Training" button to begin training the CNN.
  - Monitor the training progress and visualize the CNN architecture.

### Future Enhancements
    - Add support for validation data and metrics.
    - Extend to multi-class classification.
    - Implement additional CNN layers such as pooling and dropout.
    - Add a "Pause" and "Resume" functionality for training.