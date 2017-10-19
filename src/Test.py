import numpy
from .Dataset import MnistDataset
from .Neuralnetwork import NeuralNetwork


class TestNn:

    def __init__(self, neuronal_network, dataset, epochs):

        self.neuronal_network = neuronal_network
        self.dataset = dataset
        self.echoes = epochs

        self.output_nodes = dataset.get_length()

        # Score card to evaluate the nn
        self.scorecard = []

    def train(self):

        for e in range(self.epochs):
            # go through all records in the training data set
            for record in self.dataset.data_list:
                # split the record by the ',' commas
                all_values = record.split(',')
                # scale and shift the inputs
                inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                # create the target output values (all 0.01, except the desired label which is 0.99)
                targets = numpy.zeros(self.output_nodes) + 0.01
                # all_values[0] is the target label for this record
                targets[int(all_values[0])] = 0.99
                self.neuronal_network.train(inputs, targets)

    def query(self):

        for record in self.dataset.data_list:
            # split the record by the ',' commas
            all_values = record.split(',')
            # correct answer is first value
            correct_label = int(all_values[0])
            # scale and shift the inputs
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # query the network
            outputs = self.neuronal_network.query(inputs)
            # the index of the highest value corresponds to the label
            label = numpy.argmax(outputs)
            # append correct or incorrect to list
            if label == correct_label:
                # network's answer matches correct answer, add 1 to scorecard
                self.neuronal_network.scorecard.append(1)
            else:
                # network's answer doesn't match correct answer, add 0 to scorecard
                self.neuronal_network.scorecard.append(0)

    def get_score_rate(self):

        score_array = numpy.asarray(self.scorecard)

        return score_array.sum()/score_array.size

    def print_score_rate(self):

        print ("Score rate =", self.get_score_rate())


def main():

    # Creates the data sate
    dataset = MnistDataset("mnist_dataset/mnist_test_10.csv.txt")

    # Parameters for the Neural Network
    input_nodes = dataset.get_number_of_nodes()
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3

    # Creates the neural network
    neural_network = NeuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

    # times that te nn will train
    epochs = 5

    test = TestNn(neural_network, dataset, epochs)

    test.train()
    test.query()
    test.print_score_rate()


if __name__ == '__main__':
    main()

