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
        self.X = X
        self.y = y

        self.classes = []
        self.target = self.normalise_target()

        # Calculate input and output layer sizes
        self.input_layer_size = self.max_length_of_input_array(self.X)
        self.output_layer_size = self.max_length_of_input_array(self.target)
        self.hidden_layer_size = 0

        # Network weight matrices
        self.W1 = []
        self.W2 = []

        # Scaling factor for updating network values
        self.epsilon = 0.01

    # Utility Methods
    def max_length_of_input_array(self, input_array):
        """Returns the length of the longest input array in the training data"""

        length = 0
        for entry in input_array:
            if len(entry) > length:
                length = len(entry)
        return length

    def normalise_target(self):
        """Returns an array of boolean vector corresponding to the class of each training
        example in the form of a numpy array"""

        # Initialise target list
        target = []

        # Store all classes in a list
        for entry in self.y:
            if entry not in self.classes:
                self.classes.append(entry)

        # Convert each class into a boolean vector
        for entry in self.y:
            # Initialise a list of zeros the same length as the classes list
            target_entry = [0] * len(self.classes)
            for value in self.classes:
                # Set the index in target_entry corresponding to the training example to 1
                if entry == value:
                    target_entry[self.classes.index(value)] = 1.0
            # Append target_entry to target
            target.append(np.array(target_entry))
        return np.array(target)

    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        """Derivative of Sigmoid activiation function"""
        return np.exp(-z) / (1 + np.exp(-z))

    # Build and train network
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
            z2 = self.X.dot(self.W1)  # + b1
            a2 = self.sigmoid(z2)
            z3 = a2.dot(self.W2)  # + b2
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

        classified = []

        z2 = X.dot(self.W1)
        a2 = self.sigmoid(z2)
        z3 = a2.dot(self.W2)
        y_hat = self.sigmoid(z3)

        for result in y_hat:
            if result[0] > 0.5:
                classified.append(self.classes[0])
            elif result[1] > 0.5:
                classified.append(self.classes[1])
            elif result[2] > 0.5:
                classified.append(self.classes[2])
            else:
                classified.append('Inconclusive')

        return np.array(classified)
