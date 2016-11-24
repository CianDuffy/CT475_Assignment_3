from NeuralNetworkTester import NeuralNetworkTester

tester = NeuralNetworkTester('./Datasets/owls15.csv')
tester.test_network_and_plot_results()
tester.test_learning_rates()
