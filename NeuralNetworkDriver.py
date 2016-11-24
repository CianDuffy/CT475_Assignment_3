# Dependencies
import numpy as np
from NeuralNetworkClassifier import NeuralNetworkClassifier
from CSVFileDelegate import CSVFileDelegate


class NeuralNetworkDriver(object):
    def __init__(self, filepath):
        """
        Class responsible for training and testing the Neural Network Classifier
        :param filepath: path to the file containing the dataset
        """

        self.filepath = filepath

    def build_network_and_classify_data(self):
        """
        Builds, trains and tests the neural network ten times, tells the CSVFileDelegate to output the results to csv
        and returns the results to the GUI
        :return: Dictionary of results
        """

        final_results = []
        csv_results = []
        # Build, train and test the network 10 times
        for index in range(0, 10, 1):
            results_dictionary = {}

            # Initialise new CSVFileDelegate
            csv_delegate = CSVFileDelegate(self.filepath)

            # Initialise Neural Network Classifier with 30 hidden units and train it for 5000 iterations
            neural_network = NeuralNetworkClassifier(csv_delegate.training_data, csv_delegate.training_target)
            neural_network.build_network(30, 5000, 0.05)

            # Store results and calculate accuracy
            results = neural_network.classify_set(csv_delegate.testing_data)
            accuracy = self.compare_result_to_target(results, csv_delegate.testing_target)
            results_dictionary['input'] = csv_delegate.testing_target
            results_dictionary['result'] = results
            results_dictionary['accuracy'] = accuracy

            # Store results in format for CSV output
            for i in range(0, len(csv_delegate.testing_target), 1):
                csv_dictionary = {'input': csv_delegate.testing_target[i], 'result': results[i],
                                  'fold_number': index + 1}
                csv_results.append(csv_dictionary)

            final_results.append(results_dictionary)

        # Write results to CSV file
        csv_delegate = CSVFileDelegate(self.filepath)
        csv_delegate.write_results_to_csv(csv_results)

        # Return Dictionary of results
        return final_results

    @staticmethod
    def compare_result_to_target(result, target):
        """
        Returns the proportion similarity between two lists
        :param result: First list
        :param target: Second list
        :return: The proportion accuracy between the two lists
        """

        match_count = np.sum(np.array(result) == np.array(target))
        match_percentage = float(match_count) / len(result)
        return match_percentage
