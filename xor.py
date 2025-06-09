import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
# from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Visualization function
def visualize(model, X, y, title):
    plt.figure(figsize=(6, 6))
    plot_decision_regions(X, y, clf=model, legend=2)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Single-layer perceptron
slp = Perceptron(max_iter=1000, tol=1e-3)
slp.fit(X, y)

# Multi-layer perceptron
mlp = MLPClassifier(hidden_layer_sizes=(4,), activation='tanh', max_iter=5000, random_state=42)
mlp.fit(X, y)

# Visualize decision boundaries
print("Single-Layer Perceptron Accuracy:", slp.score(X, y))
visualize(slp, X, y, "Single-Layer Perceptron Decision Boundary")

print("Multi-Layer Perceptron Accuracy:", mlp.score(X, y))
visualize(mlp, X, y, "Multi-Layer Perceptron Decision Boundary")