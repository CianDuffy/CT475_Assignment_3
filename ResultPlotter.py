import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class ResultPlotter(object):
    def __init__(self, x_values, y_values, y_errors, iterations, baseline, filename):
        """
        Class that uses matplotlib to generate line plots based on the accuracy and exports them to .png files
        :param x_values: Number of hidden units
        :param y_values: Mean prediction accuracy
        :param y_errors: Standard error in the prediction
        :param iterations: Number of training iterations
        :param baseline: Minimum accuracy value
        :param filename: Name of the dataset file
        """

        self.x_values = x_values
        self.y_values = y_values
        self.y_errors = y_errors
        self.iterations = iterations
        self.baseline = baseline
        self.filename = filename

    def generate_combined_plot(self, x_values, y_value_sets, iterations):
        """
        Creates a plot comparing the mean accuracy values for different numbers of hidden units
        :param x_values: Number of units in the hidden layer
        :param y_value_sets: List of the mean accuracy values for each number of hidden units for each iteration number
        :param iterations: Iteration numbers
        :return: None
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for index in range(0, len(y_value_sets), 1):
            label = str(iterations[index]) + " Iterations"
            ax.plot(x_values, y_value_sets[index], label=label)

        plt.title("Neural Network Classification Accuracy vs Hidden\nLayer Dimensionality: Comparison")
        plt.xlabel("Number of units in Hidden Layer")
        plt.ylabel("Mean Classification Accuracy with Standard Error")
        plt.ylim([-0.1, 1.1])
        plt.xlim([-0.25 * max(x_values), 1.25 * max(x_values)])
        plt.grid()

        box = ax.get_position()

        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.8])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=3)

        # plt.legend(loc='upper left')
        plt.savefig("Graphs/ResultsComparison_" + self.filename + ".png", format='png')

    def generate_plot_with_errors(self):
        """
        Creates a plot of the mean prediction accuracy for each hidden layer size with error bars showing the standard
        error
        :return: None
        """
        print "Plotting Results"
        plt.figure()
        plt.errorbar(self.x_values, self.y_values, yerr=self.y_errors)
        plt.axhline(self.baseline, c="r")

        print "adding plot labels"
        plt.title("Neural Network Classification Accuracy for " + str(self.iterations) + " Iterations")
        plt.xlabel("Number of Units in Hidden Layer")
        plt.ylabel("Mean Classification Accuracy with Standard Error")

        print "Updating plot limits"
        plt.ylim([-0.1, 1.1])
        plt.xlim([0, 1.1 * max(self.x_values)])
        plt.grid()

        print "displaying plot"
        name_template = "Graphs/Results{0:02d}Iterations_" + self.filename + ".png"
        plt.savefig(name_template.format(self.iterations), format='png')

    @staticmethod
    def generate_learning_rate_plot_with_errors(x_values, y_values, y_errors):
        """
        Creates a plot of the mean prediction accuracy for eachlearning rate value with error bars showing the standard
        error
        :return: None
        """
        print "Plotting Results"
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.errorbar(x_values, y_values, yerr=y_errors)

        print "adding plot labels"
        plt.title("Neural Network Classification Accuracy")
        plt.xlabel("Learning Rate")
        plt.ylabel("Mean Classification Accuracy with Standard Error")

        print "Updating plot limits"
        plt.ylim([0, 1.1])
        ax.set_xscale('log')
        plt.xlim([0, 1.1 * max(x_values)])
        plt.grid()

        print "displaying plot"
        name_template = "Graphs/Learning_Rates.png"
        plt.savefig(name_template, format='png')
