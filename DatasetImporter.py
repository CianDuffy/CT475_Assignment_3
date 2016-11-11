import csv
import random
import numpy as np


# Utility Functions
def shuffle_and_split_list_with_ratio(input_list, ratio):
    """Shuffles and separates a list into two lists whose sizes are determined by the ratio input


    Keyword Arguments:
    input_list -- List that will be shuffled and split into two separate lists
    ratio -- proportion of the list that will be added into the first returned list
    """
    upper_list = []
    lower_list = []
    random.shuffle(input_list)
    index = 0
    cut_off = len(input_list) * ratio

    for row in input_list:
        if index < cut_off:
            upper_list.append(row)
        else:
            lower_list.append(row)
        index += 1

    return upper_list, lower_list


def normalise_list(input_list):
    """Normalises a list of floats"""
    input_array = np.array(input_list)
    return input_array / input_array.max(axis=0)


class DatasetImporter(object):

    def __init__(self, filepath):
        """Initialise DatasetImporter Object


        Keyword Arguments:
        filepath -- path to the csv file containing the data required
        """
        self.data_file = open(filepath, 'rb')
        reader = csv.reader(self.data_file)
        self.full_data_list = list(reader)

        self.training_set, self.testing_set = shuffle_and_split_list_with_ratio(self.full_data_list, 2.0 / 3.0)

        self.training_data, self.training_target = self.split_data_and_target_lists(self.training_set)
        self.testing_data, self.testing_target = self.split_data_and_target_lists(self.testing_set)

        self.training_data = normalise_list(self.training_data)
        self.testing_data = normalise_list(self.testing_data)

    @staticmethod
    def split_data_and_target_lists(input_list):
        """Splits a list of training examples into data and target lists"""
        data = []
        target = []
        for row in input_list:
            temp_data = []
            for i in range(0, len(row) - 1, 1):
                temp_data.append(row[i])
            data.append(temp_data)
            target.append(row[len(row) - 1])

        return np.array(data).astype(np.float), np.array(target)
