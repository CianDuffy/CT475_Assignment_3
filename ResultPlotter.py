import matplotlib.pyplot as plt


class ResultPlotter(object):
    def __init__(self, x_values, y_values, y_errors, iterations, baseline):
        """Creates a scatter plot with error bars


        Keyword Arguments:
        x_values -- values for x-axis
        y_values -- values for y-axis
        y_errors -- standard error in the y-axis values
        iterations -- number of training iterations used to build classifier
        baseline -- baseline accuracy for dataset
        """

        self.x_values = x_values
        self.y_values = y_values
        self.y_errors = y_errors
        self.iterations = iterations
        self.baseline = baseline

    def generate_combined_plot(self, x_values, y_value_sets, iterations):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for index in range(0, len(y_value_sets), 1):
            label = str(iterations[index]) + " Iterations"
            ax.plot(x_values, y_value_sets[index], label=label)

        plt.title("Neural Network Classification Accuracy vs Hidden\nLayer Dimensionality: Comparison")
        plt.xlabel("Number of Nodes in Hidden Layer")
        plt.ylabel("Mean Classification Accuracy with Standard Error")
        plt.ylim([-0.1, 1.1])
        plt.xlim([-0.25 * max(x_values), 1.25 * max(x_values)])
        plt.grid()

        box = ax.get_position()

        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=5)

        # plt.legend(loc='upper left')
        plt.savefig("ResultsComparison.png", format='png')

    def generate_plot_with_errors(self):
        print "Plotting Results"
        plt.figure()
        plt.errorbar(self.x_values, self.y_values, yerr=self.y_errors)
        plt.axhline(self.baseline, c="r")

        print "adding plot labels"
        plt.title("Neural Network Classification Accuracy for " + str(self.iterations) + " Iterations")
        plt.xlabel("Number of Nodes in Hidden Layer")
        plt.ylabel("Mean Classification Accuracy with Standard Error")

        print "Updating plot limits"
        plt.ylim([-0.1, 1.1])
        plt.xlim([0, 1.1 * max(self.x_values)])
        plt.grid()

        print "displaying plot"
        name_template = "Results{0:02d}Iterations.png"
        plt.savefig(name_template.format(self.iterations), format='png')
