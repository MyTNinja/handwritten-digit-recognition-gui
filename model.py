import numpy as np
import matplotlib.pyplot as plt 

import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Fix random seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape to (samples, 28, 28, 1) — channels_last format
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32')
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')

# Normalize pixels to [-1, 1]
X_train = (X_train - 127.5) / 127.5
X_test = (X_test - 127.5) / 127.5

input_shape = (28, 28, 1)
num_classes = 10

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile model with sparse categorical crossentropy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=10,
          batch_size=128)

# Select one test image
image = X_test[1]  # shape (28, 28, 1)
image_input = np.expand_dims(image, axis=0)  # shape (1, 28, 28, 1)

# Predict class
pred_probs = model.predict(image_input, verbose=0)  # shape (1, 10)
pred_class = np.argmax(pred_probs, axis=1)[0]

# Show the result
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Predicted digit: {pred_class}")
plt.axis('off')
plt.show()

# Save model
model.save("model.keras")
print("Model saved to model.keras")