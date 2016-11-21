# Dependencies
import numpy as np
import difflib


class NeuralNetworkClassifier(object):
    def __init__(self, X, y):
        """Initialise NeuralNetworkClassifier Object


        Keyword Arguments:
        X -- data array
        y -- target array
        """

        # Store data and target matrices
        self.X = self.prepare_data(X)
        self.y = y
        self.classes = self.get_classes(self.y)

        # target is converted from categorical string values into boolean vectors
        self.target = self.normalise_categorical_data(self.y)

        # Calculate input and output layer sizes
        self.input_layer_size = len(self.X[0])

        # the network's hidden layer size is initialised to 0
        self.hidden_layer_size = 0

        # The network's output layer has one unit for each class in the target data
        self.output_layer_size = len(self.classes)

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
        # Assign value of hidden layer size
        self.hidden_layer_size = hidden_layer_size

        # Initialise Weight matrices to random values.
        np.random.seed(0)
        self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size, self.output_layer_size)

        count = 0
        # Repeat for specified number of iterations
        for i in range(0, iterations, 1):

            # Forward Propagation
            z2 = self.X.dot(self.W1)
            a2 = self.sigmoid(z2)
            z3 = a2.dot(self.W2)
            y_hat = self.sigmoid(z3)

            # Print accuracy of training prediction every 500 iterations
            # if i % 500 == 0:
            #     training_accuracy = self.calculate_training_accuracy(y_hat)
            #     print "Iteration Number = " + str(i)
            #     print "Accuracy = " + str(training_accuracy)

            # Back Propagation
            delta_3 = y_hat - self.target
            delta_3 = np.multiply(delta_3, self.sigmoid_prime(z3))
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
        """
        Classify a set of testing data
        :param X: Set of testing data to be classified
        :return: Numpy array of results
        """

        data = self.prepare_data(X)

        classified = []

        # Input data is propagated forward through the artificial neural network
        z2 = data.dot(self.W1)
        a2 = self.sigmoid(z2)
        z3 = a2.dot(self.W2)
        y_hat = self.sigmoid(z3)

        # For each result in y_hat, the index with the highest value corresponds to the class to be added to the array.
        for result in y_hat:
            normalised_result = self.normalise_numerical_data(result)
            max_probability = 0
            prediction = "inconclusive"
            for index in range(0, len(result), 1):
                if normalised_result[index] > max_probability:
                    max_probability = normalised_result[index]
                    prediction = self.classes[index]
            classified.append(prediction)

        return np.array(classified)

    # Utility Methods
    def prepare_data(self, input_array):
        """
        Prepares the input data for the network.
        Categorical data is converted into boolean vector form
        Numerical data is normalised
        :param input_array: the array of training examples to be prepared
        :return: Array of correctly formatted input data
        """

        # Create array of arrays, each contains one feature of the dataset.
        separated_features = np.array(map(list, zip(*input_array)))

        # For each array:
        temp_array = []
        for feature_values in separated_features:
            # Check if data is categorical or not
            if self.target_is_categorical(feature_values):
                # If the data is categorical, convert it to a boolean vectors
                feature_values = self.normalise_categorical_data(feature_values)
            else:
                # Otherwise, the data is numerical and will be normalised
                feature_values = self.normalise_numerical_data(feature_values)
            temp_array.append(np.array(feature_values))

        reconstructed_data = []
        # Reconstruct the feature array with the new data
        for index in range(0, len(temp_array[0]), 1):
            reconstructed_entry = []
            for entry in temp_array:
                # If the entry is a boolean vector, append each entry in turn rather than the vector itself
                if isinstance(entry[index], np.ndarray):
                    for element in entry[index]:
                        reconstructed_entry.append(element)
                else:
                    reconstructed_entry.append(entry[index])
            reconstructed_data.append(np.array(reconstructed_entry))
        reconstructed_data = np.array(reconstructed_data)
        return reconstructed_data

    def normalise_categorical_data(self, input_array):
        """
        Returns an array of boolean vectors corresponding to the class of each training example in the form of a
        numpy array
        :param input_array: the array of categorical elements to be converted into boolean vectors
        :return: array of boolean vectors
        """

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
        """
        Normalises a list of floats
        :param input_array: List of floats
        :return: Normalised list of floats
        """
        """"""
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
        """
        Stores each class in an input array in a list
        :param input_array: array of categorical data to have lists checked
        :return: Array containing on instance of each class in input_array
        """

        classes = []
        for entry in input_array:
            if entry not in classes:
                classes.append(entry)
        return classes

    @staticmethod
    def sigmoid(z):
        """
        Applies a sigmoid activation function to the input
        :rtype: np.ndarray
        :param z: Matrix of floats
        :return: Matrix of floats
        """

        result = 1 / (1 + np.exp(-z))
        # Handle overflow errors
        result[np.isnan(result)] = 0
        return result

    @staticmethod
    def sigmoid_prime(z):
        """
        Applies the derivative of the sigmoid activation function to the input
        :param z: Matrix of floats
        :return: Matrix of floats
        """

        result = np.exp(-z) / ((1 + np.exp(-z)) ** 2)
        # Handle overflow errors
        result[np.isnan(result)] = 0
        return result

    @staticmethod
    def target_is_categorical(target):
        """
        Returns boolean value indicating whether or not the data is categorical
        :param target: list of
        :return: True if data in target is categorical, False otherwise
        """

        result = True
        try:
            float(target[0])
            result = False
        except ValueError:
            pass
        return result

    def calculate_training_accuracy(self, y_hat):
        """
        Returns the proportion similarity between the input list and the training target data
        :param y_hat: Array of values to be compared to training target
        :return: Float value of the accuracy of the prediction
        """

        classified = []
        # For each result in y_hat, the index with the highest value corresponds to the class to be added to the array.
        for result in y_hat:
            normalised_result = self.normalise_numerical_data(result)
            max_probability = 0.0
            prediction = "inconclusive"
            for index in xrange(len(normalised_result)):
                probability = float(normalised_result[index])
                if probability > max_probability:
                    max_probability = result[index]
                    prediction = self.classes[index]
            classified.append(prediction)

        # compare the two lists
        sequence_matcher = difflib.SequenceMatcher(None, self.y, classified)
        accuracy = sequence_matcher.ratio()
        return accuracy
