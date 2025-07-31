# Handwritten Digit Recognition GUI

Draw a digit on a canvas and get it recognized by a trained Convolutional Neural Network (CNN).  
Built using TensorFlow, Keras, Tkinter, and trained on the MNIST dataset.

---

## Features

- Interactive Tkinter GUI to draw digits
- Predicts digits (0–9) instantly using a CNN model trained on MNIST
- Clean preprocessing pipeline with normalization and resizing
- Trained model saved in `.keras` format for easy reuse
- Lightweight, runs locally without internet

---

## About MNIST dataset

This project uses the **MNIST dataset**, a standard benchmark dataset in machine learning and computer vision.  
It consists of **70,000** grayscale images of handwritten digits (0 through 9), each sized **28x28 pixels**.  
- **60,000** images for training  
- **10,000** images for testing  

The dataset is preloaded through TensorFlow/Keras APIs, enabling easy training and evaluation.

---

## Convolutional Neural Network (CNN)

The core model is a CNN designed for image classification, featuring:

- **Input Layer:** Accepts 28x28 grayscale images with a single channel
- **Convolutional Layers:** Extract spatial features using 3x3 filters with ReLU activations
- **MaxPooling Layers:** Reduce spatial dimensions to improve generalization and reduce overfitting
- **Fully Connected Dense Layers:** Learn complex relationships and patterns
- **Dropout Layer:** Adds regularization by randomly dropping neurons during training
- **Output Layer:** Uses softmax activation to classify digits into one of 10 classes

The model is compiled with `sparse_categorical_crossentropy` loss function and optimized using Adam.

---

## Structure

- **model.py**  
  Trains the CNN model on MNIST dataset, normalizes images to [-1, 1], and saves the trained model as `model.keras`.

- **main.py**  
  Implements a Tkinter GUI allowing the user to draw digits on a canvas.  
  The drawing is captured, resized to 28x28, normalized, and fed into the trained model for prediction.

- **model.keras**  
  The saved CNN model used by the GUI for inference.

- **requirements.txt**  
  Lists all required Python packages for training and running the GUI.

---

## Working

1. **Training:**  
   The CNN model is trained on the MNIST dataset images which are normalized between -1 and 1 for better convergence.

2. **Drawing:**  
   The user draws a digit on a white canvas using the mouse. The drawing is captured internally as a grayscale image.

3. **Preprocessing:**  
   The drawn image is resized to 28x28 pixels to match the MNIST format, inverted so the digit is white on black, and normalized similarly to training data.

4. **Prediction:**  
   The processed image is passed to the trained CNN model, which outputs probabilities for each digit class (0-9). The digit with the highest probability is displayed.


