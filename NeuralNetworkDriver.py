# Dependencies
import numpy as np

from NeuralNetworkClassifier import NeuralNetworkClassifier
from DatasetImporter import DatasetImporter


class NeuralNetworkDriver(object):
    def __init__(self, filepath):
        self.filepath = filepath

    def build_network_and_classify_data(self):
        final_results = []
        for index in range(0, 10, 1):
            results_dictionary = {}
            dataset_importer = DatasetImporter(self.filepath)
            neural_network = NeuralNetworkClassifier(dataset_importer.training_data, dataset_importer.training_target)
            neural_network.build_network(50, 5000)
            results = neural_network.classify_set(dataset_importer.testing_data)
            accuracy = self.compare_result_to_target(results, dataset_importer.testing_target)
            results_dictionary['input'] = dataset_importer.testing_target
            results_dictionary['result'] = results
            results_dictionary['accuracy'] = accuracy

            final_results.append(results_dictionary)

        return final_results

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
        result = True
        try:
            float(target[0])
            result = False
        except:
            pass
        return result