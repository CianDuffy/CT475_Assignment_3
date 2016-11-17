# Dependencies
import numpy as np


class NeuralNetworkClassifier(object):
    def __init__(self, X, y):
        """Initialise NeuralNetworkClassifier Object


        Keyword Arguments:
        X -- data array
        y -- target array
        """

        # Store data and target matrices
        self.X = self.prepare_data(X)
        self.classes = self.get_classes(y)
        self.target = self.normalise_categorical_data(y)

        # Calculate input and output layer sizes
        self.input_layer_size = self.max_length_of_input_array(self.X)
        self.output_layer_size = self.max_length_of_input_array(self.target)
        self.hidden_layer_size = 0

        # Network weight matrices
        self.W1 = []
        self.W2 = []

        # learning rate for updating network values
        self.epsilon = 0.01

    def build_network(self, hidden_layer_size, iterations):
        """Builds and trains a Neural Network Classifier


        Keyword Arguments:
        hidden_layer_size -- number of nodes in the hidden layer
        iterations -- number of times training data is propagated through the network
        """

        self.hidden_layer_size = hidden_layer_size
        np.random.seed(0)
        self.W1 = np.random.randn(self.input_layer_size, hidden_layer_size) / np.sqrt(self.input_layer_size)
        self.W2 = np.random.randn(hidden_layer_size, self.output_layer_size) / np.sqrt(hidden_layer_size)

        count = 0
        for i in range(0, iterations, 1):
            # Forward Propagation
            z2 = self.X.dot(self.W1)
            a2 = self.sigmoid(z2)
            z3 = a2.dot(self.W2)
            y_hat = self.sigmoid(z3)

            # Back Propagation
            delta_3 = y_hat - self.target
            delta_2 = np.dot(delta_3, self.W2.T)
            delta_2 = np.multiply(delta_2, self.sigmoid_prime(z2))

            # Update Weights
            dJdW2 = np.dot(a2.T, delta_3)
            dJdW1 = np.dot(self.X.T, delta_2)

            self.W2 -= self.epsilon * dJdW2
            self.W1 -= self.epsilon * dJdW1

            count += 1

    # Perform classification
    def classify_set(self, X):
        """Classify a set of testing data and returns a numpy array of results


        Keyword Arguments:
        X -- set of testing data to be classified
        """
        data = self.prepare_data(X)

        classified = []

        z2 = data.dot(self.W1)

        a2 = self.sigmoid(z2)
        z3 = a2.dot(self.W2)
        y_hat = self.sigmoid(z3)

        for result in y_hat:
            normalised_result = self.normalise_numerical_data(result)
            max_probability = 0
            for index in range(0, len(result), 1):
                if normalised_result[index] > max_probability:
                    max_probability = normalised_result[index]
                    prediction = self.classes[index]
            classified.append(prediction)

        return np.array(classified)

    # Utility Methods
    def prepare_data(self, input_array):
        # Create array of arrays, each contains one feature of the dataset.
        separated_features = np.array(map(list, zip(*input_array)))

        # For each array:
        temp_array = []
        for feature_values in separated_features:
            # Check if data is categorical or not
            data_type = self.get_type(feature_values[0])
            if data_type == str:
                # If the data is categorical, convert it to a boolean vectors
                feature_values = self.normalise_categorical_data(feature_values)
            else:
                feature_values = self.normalise_numerical_data(feature_values)
            temp_array.append(np.array(feature_values))

        reconstructed_data = []
        # Reconstruct the feature array with the new data
        for index in range(0, len(temp_array[0]), 1):
            reconstructed_entry = []
            for entry in temp_array:
                if isinstance(entry[index], np.ndarray):
                    for element in entry[index]:
                        reconstructed_entry.append(element)
                else:
                    reconstructed_entry.append(entry[index])
            reconstructed_data.append(np.array(reconstructed_entry))
        reconstructed_data = np.array(reconstructed_data)
        return reconstructed_data

    @staticmethod
    def get_type(value):
        tests = [
            int,
            float
        ]
        for test in tests:
            try:
                test(value)
                return test
            except:
                pass
        return str

    @staticmethod
    def max_length_of_input_array(input_array):
        """Returns the length of the longest input array in the training data"""

        length = 0
        for entry in input_array:
            if len(entry) > length:
                length = len(entry)
        return length

    def normalise_categorical_data(self, input_array):
        """Returns an array of boolean vector corresponding to the class of each training
        example in the form of a numpy array"""

        # Initialise target list
        target = []

        classes = self.get_classes(input_array)

        # Convert each class into a boolean vector
        for entry in input_array:
            # Initialise a list of zeros the same length as the classes list
            target_entry = [0] * len(classes)
            for value in classes:
                # Set the index in target_entry corresponding to the training example to 1
                if entry == value:
                    target_entry[classes.index(value)] = 1.0
            # Append target_entry to target
            target.append(np.array(target_entry))
        return np.array(target)

    @staticmethod
    def normalise_numerical_data(input_array):
        """Normalises a list of floats"""
        float_list = []
        for entry in input_array:
            float_list.append(float(entry))
        float_array = np.array(float_list)
        max_value = np.amax(float_array)
        index = 0
        return_list = []
        for entry in float_array:
            return_list.append(entry/max_value)
            index += 1
        return_array = np.array(return_list)
        return return_array

    @staticmethod
    def get_classes(input_array):
        # Store all classes in a list
        classes = []
        for entry in input_array:
            if entry not in classes:
                classes.append(entry)
        return classes

    @staticmethod
    def sigmoid(z):
        """Sigmoid activation function"""
        result = 1 / (1 + np.exp(-z))
        result[np.isnan(result)] = 0
        return result

    @staticmethod
    def sigmoid_prime(z):
        """Derivative of Sigmoid activiation function"""
        result = np.exp(-z) / (1 + np.exp(-z))
        result[np.isnan(result)] = 0
        return result


