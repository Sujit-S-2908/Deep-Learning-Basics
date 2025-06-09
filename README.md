# Deep Learning Concepts and Implementations

This repository explores key concepts in deep learning, providing implementations of fundamental techniques and models using Python. The repository includes:

1. Basic Neural Networks (`basic.py`, `basic_nn.py`, `xor.py`)
2. Feedforward Neural Networks (`dl.py`, `mnist.py`, `iris.py`)
3. Recurrent Neural Networks (RNNs) (`rnn.py`, `stock_RNN.py`)
4. Convolutional Neural Networks (CNNs) (`cnn.py`, `cnn_featuremap.py`, `fashionMNIST_pooling.py`)
5. Data Augmentation (`data_aug.py`)
6. Restricted Boltzmann Machine (`RBM.py`)
7. Sentiment Analysis (`IMDB_sentiment.py`)
8. Cityscapes Computer Vision (`cityscapes_cv.ipynb`)

## Table of Contents

-   Basic Neural Networks
-   Feedforward Neural Networks
-   Recurrent Neural Networks (RNNs)
-   Convolutional Neural Networks (CNNs)
-   Data Augmentation
-   Restricted Boltzmann Machine
-   Sentiment Analysis
-   Cityscapes Computer Vision
-   Requirements
-   Usage

## Basic Neural Networks

Custom implementations of simple neural networks built from scratch:

-   `basic.py`, `basic_nn.py`: Neuron class and small feedforward networks using sigmoid activation.
-   `xor.py`: Neural network solving the XOR problem.

## Feedforward Neural Networks

Feedforward neural networks for classification tasks:

-   `dl.py`: Classifies MNIST digits using TensorFlow/Keras.
-   `mnist.py`: Additional MNIST experiments.
-   `iris.py`: Classifies the Iris dataset.

## Recurrent Neural Networks (RNNs)

Processing sequence data with RNNs:

-   `rnn.py`: Vanilla RNN for sequence modeling.
-   `stock_RNN.py`: RNN for stock price prediction.

## Convolutional Neural Networks (CNNs)

Image processing and feature extraction with CNNs:

-   `cnn.py`: Edge detection using Sobel filter.
-   `cnn_featuremap.py`: Visualizes CNN feature maps.
-   `fashionMNIST_pooling.py`: CNN with pooling on FashionMNIST.

## Data Augmentation

-   `data_aug.py`: Demonstrates image data augmentation techniques.

## Restricted Boltzmann Machine

-   `RBM.py`: Implements a Restricted Boltzmann Machine for unsupervised learning.

## Sentiment Analysis

-   `IMDB_sentiment.py`: Sentiment analysis on the IMDB dataset using deep learning.

## Cityscapes Computer Vision

-   `cityscapes_cv.ipynb`: Computer vision tasks on the Cityscapes dataset (Jupyter Notebook).

## Requirements

Install the necessary Python libraries:

```bash
pip install numpy tensorflow scikit-image matplotlib keras scikit-learn pandas
```

## Usage

Clone the repository and run the scripts as needed:

1. Run basic neural network examples:

```bash
python basic.py
python basic_nn.py
python xor.py
```

2. Train and evaluate feedforward neural networks:

```bash
python dl.py
python mnist.py
python iris.py
```

3. Visualize and experiment with CNNs:

```bash
python cnn.py
python cnn_featuremap.py
python fashionMNIST_pooling.py
```

4. Data augmentation:

```bash
python data_aug.py
```

5. Run RNN examples:

```bash
python rnn.py
python stock_RNN.py
```

6. Restricted Boltzmann Machine:

```bash
python RBM.py
```

7. Sentiment analysis:

```bash
python IMDB_sentiment.py
```

8. Cityscapes Computer Vision (Jupyter Notebook):
   Open `cityscapes_cv.ipynb` in Jupyter Notebook or VS Code.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
