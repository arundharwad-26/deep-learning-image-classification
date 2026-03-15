# Task 2 - Deep Learning Project
# CNN for Image Classification (MNIST)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Dataset
# -----------------------------
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -----------------------------
# 2. Normalize Data
# -----------------------------
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape for CNN (add channel dimension)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# -----------------------------
# 3. Build CNN Model
# -----------------------------
# model = keras.Sequential([
#     layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
#     layers.MaxPooling2D((2,2)),
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.MaxPooling2D((2,2)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])
model = keras.Sequential([
    keras.Input(shape=(28,28,1)),
    layers.Conv2D(32,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')
])

# -----------------------------
# 4. Compile Model
# -----------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# 5. Train Model
# -----------------------------
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# -----------------------------
# 6. Evaluate Model
# -----------------------------
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# -----------------------------
# 7. Plot Accuracy Graph
# -----------------------------
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
