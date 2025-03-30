# Pokémon Type Classifier with CNN

This project implements a deep learning-based Pokémon type classifier using a Convolutional Neural Network (CNN) in TensorFlow, integrated with a Tkinter GUI for interactive training and visualization.

## Features:
- **Image Preprocessing**: Loads and processes Pokémon images, classifying them into 'grass' or 'fire' types based on color analysis (red vs. green pixels).
- **CNN Model**: A simple CNN architecture is used for binary classification.
- **Real-time Visualization**: Displays graphs of training loss and accuracy, as well as image from the dataset.
- **Training Control**: Users can specify custom epochs and learning rates for training the model.
- **Data Augmentation**: Implements basic augmentation (e.g., horizontal flipping and rotation) to improve model generalization.
