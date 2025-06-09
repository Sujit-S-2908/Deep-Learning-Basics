import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import regularizers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize

# Define models with L1 and L2 regularization
def create_model(regularizer):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu', kernel_regularizer=regularizer),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train models
l1_model = create_model(regularizers.l1(0.0001))
l2_model = create_model(regularizers.l2(0.0001))

l1_history = l1_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=1)
l2_history = l2_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=1)

test_loss, test_acc = l1_model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

test_loss, test_acc = l2_model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# Plot validation loss
plt.plot(l1_history.history['val_loss'], label='L1 Regularization')
plt.plot(l2_history.history['val_loss'], label='L2 Regularization')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.title('Comparison of L1 and L2 Regularization on MNIST')
plt.show()