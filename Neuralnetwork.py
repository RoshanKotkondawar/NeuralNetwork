import numpy
import scipy.special


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):

        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # Weight matrix initialization
        """
        self.wih = (numpy.random.rand(hidden_nodes, input_nodes) - 0.5)   # Matrix of input and hidden nodes
        self.who = (numpy.random.rand(output_nodes, hidden_nodes) - 0.5)  # Matrix of input and hidden nodes
        """
        # Weight matrix initialization (more sophisticated way)

        # Matrix of input and hidden nodes
        self.wih = numpy.random.norm(0.0, pow(self.hidden_nodes, -0.5), (self.input_nodes, -0.5))

        # Matrix of input and hidden nodes
        self.who = numpy.random.norm(0.0, pow(self.output_nodes, -0.5), (self.hidden_nodes, -0.5))

        # Activation function (sigmoid)
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self):
        pass

    def query(self, input_values):

        # Convert input values to 2d array
        inputs = numpy.array(input_values, ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)

        # Calculate the hidden outputs

        hidden_outputs = self.actvitation_function(hidden_inputs)

        # calculate signals into output layer

        output_inputs = numpy.dot(self.who, hidden_outputs)

        # calculate the final outputs

        final_outputs = self.activation_function(output_inputs)

        return final_outputs







