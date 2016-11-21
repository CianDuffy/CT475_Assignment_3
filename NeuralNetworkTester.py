# Dependencies
import ntpath

import numpy as np
import scipy.stats
from CSVFileDelegate import CSVFileDelegate
from NeuralNetworkClassifier import NeuralNetworkClassifier
from ResultPlotter import ResultPlotter


class NeuralNetworkTester(object):
    def __init__(self, filepath):
        self.filepath = filepath

        self.training_iterations_list = range(1000, 11000, 1000)
        self.hidden_layer_sizes = range(5, 55, 5)

        self.baseline_accuracy = self.calculate_accuracy_baseline(self.filepath)
        self.test_network_and_plot_results()

    def test_network_and_plot_results(self):
        accuracy_results = []
        for training_iterations in self.training_iterations_list:
            print "Testing network for " + str(training_iterations) + " iterations"

            # Initialise lists to store mean accuracy and standard error values
            mean_accuracies = []
            standard_errors = []

            # Iterates through the hidden layer sizes specified
            for hidden_layer_size in self.hidden_layer_sizes:
                # initialise list to store accuracy results
                accuracy_cache = []

                print "Testing network with hidden layer size: " + str(hidden_layer_size)
                # The Neural network is trained and tested for the specified number of iterations for each
                # hidden layer size value
                for i in range(0, 10, 1):
                    # Reinitialise dataset importer object
                    csv_delegate = CSVFileDelegate(self.filepath)

                    # Initialise Neural Network Classifier with data and target from data importer
                    neural_network_classifier = NeuralNetworkClassifier(csv_delegate.training_data,
                                                                        csv_delegate.training_target)

                    # Build and train network with <hidden_layer_size> nodes in the hidden layer
                    neural_network_classifier.build_network(hidden_layer_size, training_iterations)

                    # Use classifier to classify testing data
                    results = neural_network_classifier.classify_set(csv_delegate.testing_data)

                    # Compare results to testing target
                    accuracy = self.compare_result_to_target(results, csv_delegate.testing_target)
                    accuracy_cache.append(accuracy)
                    print accuracy

                # Store the mean and standard error values for each set of results
                mean_accuracies.append((float(sum(accuracy_cache)) / 10))
                standard_errors.append(scipy.stats.sem(accuracy_cache))

            # Plot accuracy vs number of hidden nodes with the standard error
            plotter = ResultPlotter(self.hidden_layer_sizes, mean_accuracies, standard_errors, training_iterations,
                                    self.baseline_accuracy, ntpath.basename(self.filepath))
            plotter.generate_plot_with_errors()
            accuracy_results.append(mean_accuracies)
        if plotter:
            plotter.generate_combined_plot(self.hidden_layer_sizes, accuracy_results, self.training_iterations_list)

    # Utility Methods
    @staticmethod
    def calculate_accuracy_baseline(filename):
        """Calculates the baseline accuracy for the dataset in question"""
        # Initialises DatasetImporterObject
        csv_delegate = CSVFileDelegate(filename)

        # Separates the target classes from the input data
        data, target = csv_delegate.split_data_and_target_lists(csv_delegate.full_data_list)

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

    def compare_result_to_target(self, result, target):
        """Returns the percentage similarity between two lists"""
        # result, target = lists to be compared
        if self.target_is_categorical(target):
            match_count = np.sum(np.array(result) == np.array(target))
            match_percentage = float(match_count) / len(result)
        else:
            match_count = 0.0
            for index in range(0, len(result), 1):
                if abs(float(result[index]) - float(target[index])) < 1.0:
                    match_count += 1.0
            match_percentage = match_count / len(result)
        return match_percentage

    @staticmethod
    def target_is_categorical(target):
        """Returns true the data is categorical rather than numerical"""
        result = True
        # noinspection PyBroadException
        try:
            float(target[0])
            result = False
        except:
            pass
        return result
