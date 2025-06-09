import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

# Load Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize dataset
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# One-hot encode labels
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# Display sample images from the dataset
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("Sample Images from Fashion-MNIST", fontsize=14)

for i in range(10):
    ax = axes[i // 5, i % 5]
    ax.imshow(x_train[i].reshape(28, 28), cmap='gray')
    ax.axis('off')

plt.show()

# Define CNN model with Max Pooling
def create_maxpool_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define CNN model with Average Pooling
def create_avgpool_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        AveragePooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        AveragePooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate both models
models = {"Max Pooling": create_maxpool_cnn(), "Average Pooling": create_avgpool_cnn()}
results = {}
trained_models = {}

for name, model in models.items():
    print(f"Training {name} model...")
    model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test), verbose=1)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    results[name] = test_acc
    trained_models[name] = model
    print(f"Test Accuracy with {name}: {test_acc:.4f}\n")

# Compare performance
plt.bar(results.keys(), results.values(), color=['blue', 'orange'])
plt.ylabel("Accuracy")
plt.title("Max Pooling vs. Average Pooling Performance")
plt.show()

# Display some test images with predictions
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("Predictions after Training", fontsize=14)

sample_images = x_test[:10]
true_labels = np.argmax(y_test[:10], axis=1)

for i in range(10):
    ax = axes[i // 5, i % 5]
    ax.imshow(sample_images[i].reshape(28, 28), cmap='gray')
    ax.axis('off')
    pred_maxpool = np.argmax(trained_models["Max Pooling"].predict(sample_images[i:i+1]))
    pred_avgpool = np.argmax(trained_models["Average Pooling"].predict(sample_images[i:i+1]))
    ax.set_title(f"True: {true_labels[i]}\nMax: {pred_maxpool}, Avg: {pred_avgpool}")

plt.show()
