import numpy
import matplotlib.pyplot
%matplotlib inline

class MnistDataset:

    def __init__(self, folder_path):

        data_file = open(folder_path)
        self.data_list = data_file.readlines()
        data_file.close()

    def print_data(self):

        for record in self.data_list :
            all_values = record.split(',')
            current_value = all_values[0]
            image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
            matplotlib.pyplot.imshow(image_array,cmap='Greys',interpolation='None')

            print ('Current value: %s' % current_value)


    def print_data (self, data_location) :

        all_values = self.data_list[data_location].split(',')
        current_value = all_values[data_location]
        image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
        matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')

        print('Current value: %s' % current_value)

    def get_lines_number(self): return len(self.data_list)

    def get_number_of_nodes(self):

        number_of_nodes = 0

        for record in self.data_list :
            all_values = record.split(',')
            number_of_nodes += len(all_values)

        return number_of_nodes - 1