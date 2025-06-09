import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    def total(self,input):
        return sigmoid(np.dot(self.weights,input)+self.bias)

class OurNeuralNetwork:
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.total(x)
        out_h2 = self.h2.total(x)

        out_o1 = self.o1.total(np.array([out_h1, out_h2]))

        return out_o1

network = OurNeuralNetwork()
x = np.array([2, 3]) # input
print(network.feedforward(x)) 

