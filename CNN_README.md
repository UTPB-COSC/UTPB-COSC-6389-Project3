# CNN Visualization and Training GUI

This project provides a graphical user interface (GUI) to visualize the internal workings of a Convolutional Neural Network (CNN) trained on the MNIST dataset. It shows intermediate layer outputs, final classifications, and updates these visuals in real-time as the network trains.

## Features

- **Input Image Display:**  
  View the raw 28x28 input image (scaled up for visibility) currently being processed.

- **Convolution Layer Visualization:**  
  See how the input is transformed by multiple convolution filters arranged in a grid. As training progresses, these filters and their activations become more meaningful.

- **Intermediate Layer Representations (Placeholders):**  
  Illustrations for ReLU/Pool and Flatten steps help conceptualize dimension reductions, thresholding out negative values, and vectorizing feature maps.

- **Fully Connected (FC) Layer Neurons:**  
  A vertical column of 10 neurons (for MNIST’s 10 classes) displays color intensity based on activation values, providing insight into how confident the model is about each class before softmax normalization.

- **Softmax Output and Probability Bar Chart:**  
  After the FC layer, a softmax distribution determines class probabilities. A bar chart and top-k class listings display the model’s predictions and uncertainty. Predicted class and correctness are highlighted with colored outlines.

- **Training Controls and Progress:**  
  - **Epochs and Batch Size Inputs:** Enter desired training epochs and batch size directly in the GUI.
  - **Buttons:**
    - **Start Training:** Begins the training loop, updating model weights and GUI visualization batch-by-batch.
    - **Pause/Resume:** Pauses the training steps without closing the window.
    - **Stop:** Ends training at any point.
  - **Progress Label:** Displays the current epoch, batch, and overall progress.

- **Scrolling and Large Canvas Support:**  
  The canvas is scrollable both horizontally and vertically, allowing you to explore all layers and visual elements even if the diagram grows beyond the initial window size.

## File Structure

- **src/data.py:**  
  Loads and normalizes the MNIST dataset (X_train, y_train, X_test, y_test).

- **src/network.py:**  
  Implements the CNN architecture (Conv, ReLU, Pool, Flatten, FC, Softmax) and methods for forward passes and loss computation/backpropagation.

- **src/layout.py:**  
  Defines the GUI’s static layout: positions and placeholders for input image, conv filters, intermediate layers, FC neurons, and softmax probabilities. No data or dynamic updates occur here—only the initial diagram structure.

- **src/dynamic.py:**  
  Handles dynamic updates of the GUI elements as the model changes:
  - Updates input image, conv filter activations, FC neuron color intensities, softmax bars, top-k classes, and correctness highlights after each training batch.
- **console.py:**
  - was the former main.py file
  - starts the training process and displays the progress 
- **main.py:**  
  Integrates everything:
  - Creates the Tkinter window, scrollbars, and canvas.
  - Loads the data and model.
  - Calls layout’s `draw_base_diagram()` to set initial placeholders.
  - Uses dynamic’s `update_values()` to show intermediate outputs after training steps.
  - Provides GUI controls (entry boxes for epochs/batch size, Start/Pause/Stop buttons).
  - Runs a training loop batch-by-batch, updating the GUI in real-time using `root.after()` calls.

## Usage Instructions

1. **Prerequisites:**
   - Python 3.x
   - `tkinter` (usually bundled with Python)
   - `Pillow` for image handling (`pip install Pillow`)
   - `numpy` for numerical operations (`pip install numpy`)

2. **Running the Program:**
   - Place `main.py`, `layout.py`, `dynamic.py`, `data.py`, and `network.py` in appropriate directories (e.g., `src/` for code).
   - Ensure that the MNIST dataset files (`train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`) are in a `data/` folder so `data.py` can load them.
   - Run `python main.py`.

3. **Using the GUI:**
   - **Set Epochs and Batch Size:** Enter desired values in the top-left entries before starting training.
   - **Start Training:** Click "Start Training" to begin. The diagram updates after each batch.
   - **Pause/Resume:** Click "Pause" to halt the batch updates. Click again (now "Resume") to continue.
   - **Stop:** Click "Stop" to end training at any time.
   - **Scrolling:** Use scrollbars if some parts of the diagram are off-screen.

4. **Interpreting the Visualization:**
   - **Input:** Shows the current training image being processed.
   - **Conv Filters:** Grayscale mini-images show how filters respond to features. As training progresses, patterns should become clearer.
   - **ReLU/Pool/Flatten Placeholders:** Conceptual visuals (arrows, boxes, dots) to remind you how data is transformed.
   - **FC Neurons:** Their color intensity reflects activation strength. The predicted class neuron might brighten or stand out over time as model learns.
   - **Softmax Bars and Top-k Classes:** The bars grow or shrink to reflect class probabilities. Top classes and predicted vs. label information show model accuracy evolving.

5. **Ending the Session:**
   - Once training completes or you stop it, you can close the window. The final state of the model and diagram remain visible until the program closes.

