# Deep Learning Concepts and Implementations
This repository explores key concepts in deep learning, providing implementations of fundamental techniques and models using Python. The repository includes:

1. Basic Neural Networks
2. Feedforward Neural Networks
3. Recurrent Neural Networks (RNNs)
4. Convolutional Neural Networks (CNNs)
## Table of Contents
- Basic Neural Networks
- Feedforward Neural Networks
- Recurrent Neural Networks (RNNs)
- Convolutional Neural Networks (CNNs)
- Requirements
- Usage
## Basic Neural Networks
The basic.py script provides a custom implementation of a simple neural network built from scratch, using the following components:

- Neuron class for individual neuron computations using the sigmoid activation function.
- A small feedforward neural network that processes two inputs and outputs a single prediction. 
## Feedforward Neural Networks
The dl.py script demonstrates a feedforward neural network using TensorFlow and Keras to classify handwritten digits from the MNIST dataset:

- Architecture: Input layer (Flatten), one hidden layer (128 neurons with ReLU), and an output layer (10 neurons with Softmax).
- Training: Categorical Crossentropy loss and Adam optimizer.
- Evaluation: Accuracy on the test dataset.
### Key Results:
- Model achieves high accuracy on the MNIST dataset.
## Recurrent Neural Networks (RNNs)
The RNN implementation demonstrates how sequence data can be processed using a simple RNN architecture. The RNN is particularly useful for tasks involving sequential or time-series data, such as text, stock price predictions, or language modeling.

### Highlights:
- Architecture: Vanilla RNN, which processes sequential data step-by-step and updates the hidden state.
- Applications: Suitable for tasks involving temporal dependencies.
## Convolutional Neural Networks (CNNs)
The cnn.py script highlights the power of Convolutional Neural Networks through the application of the Sobel filter:

- Task: Edge detection in images using the horizontal Sobel filter.
- Tools: NumPy for convolution, skimage for image processing, and Matplotlib for visualization.
### Example Output:
Displays the original image alongside the filtered image, highlighting horizontal edges.

## Requirements
1. Install the necessary Python libraries:

```bash
pip install numpy tensorflow scikit-image matplotlib
````

## Usage
Clone the repository and run the scripts as needed:

1. Run the basic neural network:
````bash
python basic.py
````

2. Train and evaluate the feedforward neural network:
````bash
python dl.py
````
 
3. Visualize CNN edge detection:
````bash
python cnn.py
```` 
4. Implement RNN:
````bash
python rnn.py
````
## License
This project is licensed under the MIT License. See the LICENSE file for details.
