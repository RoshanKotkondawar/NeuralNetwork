import numpy


class TestNn:

    def __init__(self, neuronal_network, dataset, epochs):

        self.neuronal_network = neuronal_network
        self.dataset = dataset
        self.echoes = epochs

        self.output_nodes = dataset.get_length()

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
                self.neuronal_network.scorecard.append(1)





