import numpy

def tanh(x):
    return numpy.tanh(x)

def tanh_derivertive(x):
    tanh_ = tanh(x)
    return 1.0 - tan_ * tan_

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def sigmoid_derivertive(x):
    sig = sigmoid(x)
    return sig * (1 - sig)