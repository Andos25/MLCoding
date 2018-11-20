import numpy
import random
from formula import *

class Network:
    def __init__(self, sizes):
        self.number_layers = len(sizes)
        self.biases = [numpy.random.rand(n , 1) for n in sizes]
        self.weights = [numpy.random.rand(n, m) for n, m in zip(sizes[:-1], sizes[1:])]
        self.output = [numpy.random.rand(n, m) for n, m in zip(sizes[:-1], sizes[1:])]

    def sigmoid(self, x):
        return 1.0/(1 + numpy.exp(-x))

    def backprop(self, x, y):
        grad_biases = [numpy.zeros(b.shape) for b in self.biases]
        grad_weights = [numpy.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        wx_b_list = list()

        for w, b in zip(self.weights, self.biases):
            wx_b = numpy.dot(activation, w) + b
            wx_b_list.append(wx_b)
            activation = sigmoid(wx_b)
            activations.append(activation)

        g = (activations[-1] - y) * activations[-1] * (1 - activations[-1])
        grad_biases[-1] = g
        grad_weights[-1] = numpy.dot(g, activation[-2])

        for i in range(2, self.number_layers):
            g = numpy.dot(g, self.weights[-i + 1]) * activations[-i] * (1 - activations[-i])
            grad_biases[-i] = g
            grad_weights[-i] = numpy.dot(g, activations[-i-1])

        return grad_weights, grad_biases

    def update_mini_batch(self, mini_batch, lr):
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(lr/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(lr/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def SGD(self, training_data, epochs, mini_batch_size, lr):
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)

    