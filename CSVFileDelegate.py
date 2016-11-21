import csv
import ntpath
import random
import numpy as np


# Utility Functions
def shuffle_and_split_list_with_ratio(input_list, ratio):
    """
    Shuffles and separates a list into two lists whose sizes are determined by the ratio input
    :param input_list: List that will be shuffled and split into two separate lists
    :param ratio: Proportion of the list that will be added into the first returned list
    :return: Two lists with <ratio> of the data contained in the first and the remainder in the second
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


class CSVFileDelegate(object):

    def __init__(self, filepath):
        """Initialise DatasetImporter Object


        Keyword Arguments:
        filepath -- path to the csv file containing the data required
        """
        self.filepath = filepath

        self.data_file = open(filepath, 'rb')
        reader = csv.reader(self.data_file)
        self.full_data_list = list(reader)

        self.training_set, self.testing_set = shuffle_and_split_list_with_ratio(self.full_data_list, 2.0 / 3.0)

        self.training_data, self.training_target = self.split_data_and_target_lists(self.training_set)
        self.testing_data, self.testing_target = self.split_data_and_target_lists(self.testing_set)

    def write_results_to_csv(self, results):
        """Writes the results of the classification to a csv file.
        the csv file's name follows the format: 'output_<filename>.csv'

        Keyword Arguments:
        results -- a dictionary containing the iteration number, target and results from the classification operation
        """
        filename = ntpath.basename(self.filepath)

        with open("output_" + filename, "wb") as output_file:
            keys = results[0].keys()
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)

    @staticmethod
    def split_data_and_target_lists(input_list):
        """
        Splits a list of training examples into data and target lists
        :param input_list: list of training examples to be split into data and target lists
        :return: two numpy arrays, one containing the training examples, one containing the target result
        """

        data = []
        target = []
        for row in input_list:
            temp_data = []
            for i in range(0, len(row) - 1, 1):
                temp_data.append(row[i])
            data.append(temp_data)
            target.append(row[len(row) - 1])

        return np.array(data), np.array(target)