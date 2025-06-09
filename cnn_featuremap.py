import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# Define CNN model with Batch Normalization and Dropout
def create_cnn(kernel_size=3):
    model = Sequential([
        Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same', input_shape=(32, 32, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate models with different kernel sizes
kernel_sizes = [3, 5, 7]
history = {}

for kernel_size in kernel_sizes:
    print(f"Training CNN with kernel size {kernel_size}x{kernel_size}")
    model = create_cnn(kernel_size)
    history[kernel_size] = model.fit(x_train, y_train, epochs=15, batch_size=128, validation_data=(x_test, y_test), verbose=1)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy with {kernel_size}x{kernel_size} kernel: {test_acc:.4f}\n")

# Select an image for visualization
sample_image = np.expand_dims(x_train[0], axis=0)

# Extract feature maps
feature_maps = {}
for kernel_size in kernel_sizes:
    inputs = Input(shape=(32, 32, 3))
    conv_layer = Conv2D(16, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(inputs)
    model = Model(inputs=inputs, outputs=conv_layer)
    feature_maps[kernel_size] = model.predict(sample_image)

# Plot feature maps with enhanced clarity
fig, axes = plt.subplots(len(kernel_sizes), 16, figsize=(20, 10))
fig.suptitle("Feature Maps with Different Kernel Sizes", fontsize=16)

for i, kernel_size in enumerate(kernel_sizes):
    for j in range(16):
        axes[i, j].imshow(feature_maps[kernel_size][0, :, :, j], cmap='viridis')
        axes[i, j].axis('off')
    axes[i, 0].set_ylabel(f"Kernel: {kernel_size}x{kernel_size}", fontsize=14)

plt.show()