import numpy

class Network:
    def __init__(self, input_size, hidden_size, output_size):
        self.first_layer = numpy.random.rand(input_size, hidden_size)
        self.second_layer = numpy.random.rand(hidden_size, output_size)
        self.theta_list = numpy.zeros(1, output_size)
        self.gamma_list = numpy.zeros(1, hidden_size)

    def sigmoid(x):
        return 1.0/(1 + numpy.exp(-x))

    def BP_train(self, x_list, y_list, learning_rate):
        hidden_output_list = x_list.dot(self.first_layer)
        predict_output_list = hidden_output_list.dot(self.second_layer)
        g = predict_label_list * (1 - predict_label_list) * (y_list - predict_label_list)
        e = hidden_output_list * (1 - hidden_output_list) * (self.second_layer.dot(g))
        delta_w = learning_rate * hidden_output_list.dot(g)
        delta_theta_list = - learning_rate * g
        delta_v = learning_rate * x_list.dot(e)
        delta_gamma_list = - learning_rate * e

        self.second_layer += delta_w
        self.first_layer += delta_v
        self.theta_list += delta_theta_list
        self.gamma_list += delta_gamma_list

    