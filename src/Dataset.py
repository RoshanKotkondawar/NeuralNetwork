import numpy
import matplotlib.pyplot
%matplotlib inline

class MnistDataset:

    def __init__(self, folder_path):
        data_file = open(folder_path)
        self.data_list = data_file.readlines()
        data_file.close()



    def get_length(self): return len(self.data_list)
