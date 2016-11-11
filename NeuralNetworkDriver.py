# Dependencies
import numpy as np
import scipy.stats
from DatasetImporter import DatasetImporter
from NeuralNetworkClassifier import NeuralNetworkClassifier
from ResultPlotter import ResultPlotter

# Define the hidden layer sizes, the number of iterations for each size and the number of training iterations
testing_iterations = 10
training_iterations_list = [100, 200, 300, 400, 500]  # , 1000, 2000, 5000, 10000]
hidden_layer_sizes = range(2, 11, 1)


# Utility Methods
def compare_result_to_target(result, target):
    """Returns the percentage similarity between two lists"""
    # result, target = lists to be compared
    match_count = np.sum(np.array(result) == np.array(target))
    match_percentage = float(match_count) / len(result)
    return match_percentage


def calculate_accuracy_baseline():
    """Calculates the baseline accuracy for the dataset in question"""
    # Initialises DatasetImporterObject
    dataset_importer = DatasetImporter("owls15.csv")

    # Separates the target classes from the input data
    data, target = dataset_importer.split_data_and_target_lists(dataset_importer.full_data_list)

    target_classes = {}
    for entry in target:
        if entry not in target_classes:
            target_classes[entry] = 1
        else:
            target_classes[entry] += 1
    max_count = 0

    for target_class in target_classes:
        if target_classes[target_class] > max_count:
            max_count = target_classes[target_class]

    percentage = float(max_count) / float(len(target))

    return percentage

# Calculate the baseline accuracy for the dataset
baseline_accuracy = calculate_accuracy_baseline()

accuracy_results = []
for training_iterations in training_iterations_list:
    print "Testing network for " + str(training_iterations) + " iterations"

    # Initialise lists to store mean accuracy and standard error values
    mean_accuracies = []
    standard_errors = []

    # Iterates through the hidden layer sizes specified
    for hidden_layer_size in hidden_layer_sizes:
        # initialise list to store accuracy results
        accuracy_cache = []

        print "Testing network with hidden layer size: " + str(hidden_layer_size)
        # The Neural network is trained and tested for the specified number of iterations for each
        # hidden layer size value
        for i in range(0, testing_iterations, 1):
            # Reinitialise dataset importer object
            datasetImporter = DatasetImporter('owls15.csv')

            # Initialise Neural Network Classifier with data and target from data importer
            neuralNetworkClassifier = NeuralNetworkClassifier(datasetImporter.training_data,
                                                              datasetImporter.training_target)

            # Build and train network with <hidden_layer_size> nodes in the hidden layer
            neuralNetworkClassifier.build_network(hidden_layer_size, training_iterations)

            # Use classifier to classify testing data
            results = neuralNetworkClassifier.classify_set(datasetImporter.testing_data)

            # Compare results to testing target
            accuracy = compare_result_to_target(results, datasetImporter.testing_target)
            accuracy_cache.append(accuracy)

        # Store the mean and standard error values for each set of results
        mean_accuracies.append((float(sum(accuracy_cache)) / 10))
        standard_errors.append(scipy.stats.sem(accuracy_cache))

    # Plot accuracy vs number of hidden nodes with the standard error
    plotter = ResultPlotter(hidden_layer_sizes, mean_accuracies, standard_errors, training_iterations, baseline_accuracy)
    # plotter.generate_plot_with_errors()
    accuracy_results.append(mean_accuracies)

plotter.generate_combined_plot(hidden_layer_sizes, accuracy_results, training_iterations_list)
