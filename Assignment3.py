
# Dependencies
import csv
import random
import numpy as np

# Initialise training and testing lists
training_set = []
training_data = []
training_target = []
testing_set = []
testing_data = []
testing_target = []

# Randomly separates a list into two lists whose sizes are determined by the ratio input
def randomize_and_split_list_with_ratio(input_list, ratio):
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

# Splits a list of training examples into data and target lists
def split_data_and_target_lists(input_list):
    data = []
    target = []
    for row in input_list:
        temp_data = []
        for i in range(0, len(row)-1, 1):
            temp_data.append(row[i])
        data.append(temp_data)
        target.append(row[len(row)-1])

    return data, target

# Import owls dataset from owls15.csv into a list
data_file = open('owls15.csv', 'rb')
reader = csv.reader(data_file)
owl_list = list(reader)

# List is shuffled. The first 2/3 of the data set is placed into the training
# set and the remainder is added into the testing data set
training_set, testing_set = randomize_and_split_list_with_ratio(owl_list, 2.0 / 3.0)

# Training and testing sets are each split into data and target lists
testing_data, testing_target = split_data_and_target_lists(testing_set)
training_data, training_target = split_data_and_target_lists(training_set)