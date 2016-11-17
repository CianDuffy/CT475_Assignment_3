import Tkinter
from tkFileDialog import askopenfilename
import ntpath
import numpy as np
from NeuralNetworkDriver import NeuralNetworkDriver
from NeuralNetworkTester import NeuralNetworkTester


class NeuralNetworkGUI(object):
    def __init__(self):
        # Initialise main window
        self.main_window = Tkinter.Tk()
        self.main_window.wm_title("Artificial Neural Network")
        self.main_window.resizable(width=False, height=False)
        self.initial_width = 700
        self.initial_height = 500
        self.main_window.geometry('{}x{}'.format(self.initial_width, self.initial_height))

        # Initialise colours
        self.gallery_colour = "#EFEFEF"
        # Add canvas for instruction text
        self.text_canvas = Tkinter.Canvas(self.main_window, width=self.initial_width, height=self.initial_height - 5)
        self.centre_label = self.text_canvas.create_text(self.initial_width / 2, self.initial_height / 2, anchor="c",
                                                         font="Helvetica", text='Import your dataset to get started')
        self.text_canvas.place(relx=0.5, rely=-0.1, anchor="n")
        self.text_canvas.configure(background=self.gallery_colour)

        self.import_button = Tkinter.Button(self.main_window, text="Import Dataset...",
                                            command=self.on_click_import_button)
        self.import_button.place(relx=0.5, rely=0.6, anchor="c")

        self.filename = ""
        self.filepath = ""
        self.hidden_layer_sizes = range(10, 51, 10)
        self.training_iterations = range(1000, 10001, 1000)

        self.main_window.mainloop()

    def on_click_import_button(self):
        opts = {'filetypes': [('CSV files', '.csv')]}
        self.filepath = askopenfilename(**opts)  # show an "Open" dialog box and return the path to the selected file
        self.filename = self.get_filename_from_path(self.filepath)
        self.update_centre_label(self.filename)

        self.import_button.destroy()
        float_width = float(self.initial_width)
        float_height = float(self.initial_height)

        bottom_buttons_y_position = (float_height - 20.0) / float_height
        reset_button_x_position = 20.0 / float_width
        test_button_x_position = (float_width - 20.0) / float_width

        self.reset_button = Tkinter.Button(self.main_window, text="Reset", command=self.on_click_reset_button)
        self.reset_button.place(relx=reset_button_x_position, rely=bottom_buttons_y_position, anchor="sw")

        self.classify_button = Tkinter.Button(self.main_window, text=("Classify " + self.filename),
                                              command=self.on_click_classify_button)
        self.classify_button.place(relx=0.5, rely=bottom_buttons_y_position, anchor="s")

        self.test_button = Tkinter.Button(self.main_window, text=("Test " + self.filename),
                                          command=self.on_click_test_button)
        self.test_button.place(relx=test_button_x_position, rely=bottom_buttons_y_position, anchor="se")

        self.main_window.mainloop()

    def on_click_classify_button(self):
        self.update_centre_label("Classifying")
        driver = NeuralNetworkDriver(self.filepath)
        final_results = driver.build_network_and_classify_data()
        print final_results

        accuracy_sum = 0
        index = 1
        for result_dictionary in final_results:
            self.print_classification_results(result_dictionary, index)
            accuracy_sum += float(result_dictionary['accuracy'])

        mean_accuracy = accuracy_sum / len(final_results)
        self.update_centre_label("Classification complete.\nMean accuracy: " + str(round(100 * mean_accuracy, 2)) + "%")
        self.main_window.mainloop()

    def on_click_test_button(self):
        self.update_centre_label("Testing")
        tester = NeuralNetworkTester(self.filepath)
        tester.test_network_and_plot_results()

    def on_click_reset_button(self):
        self.reset_button.destroy()
        self.classify_button.destroy()
        self.test_button.destroy()

        self.filename = ""
        self.filepath = ""

        self.import_button = Tkinter.Button(self.main_window, text="Import Dataset...",
                                            command=self.on_click_import_button)
        self.import_button.place(relx=0.5, rely=0.6, anchor="c")

        self.centre_label.configure(text="Import your dataset to get started")
        self.main_window.mainloop()

    @staticmethod
    def get_filename_from_path(filepath):
        return ntpath.basename(filepath)

    def print_classification_results(self, result_dictionary, index):
        input_array = result_dictionary['input']
        result_array = result_dictionary['result']
        accuracy = float(result_dictionary['accuracy'])
        print "----------------------------------------------------------"
        print "Iteration #" + str(index)
        print "Input list:"
        print input_array
        print "Predicted list:"
        print result_array
        print "Matched Entries:"
        print self.matched_result_list_for_input_lists(input_array, result_array)
        print "Accuracy: " + str(round(100 * accuracy, 2)) + "%"

    @staticmethod
    def matched_result_list_for_input_lists(input, result):
        matched_list = []
        for index in range(0, len(input), 1):
            if input[index] == result[index]:
                matched_list.append("Match")
            else:
                matched_list.append("Fail")

        return np.array(matched_list)

    def update_centre_label(self, label_text):
        self.text_canvas.destroy()
        self.text_canvas = Tkinter.Canvas(self.main_window, width=self.initial_width, height=self.initial_height - 5)
        self.text_canvas.place(relx=0.5, rely=-0.1, anchor="n")
        self.text_canvas.configure(background=self.gallery_colour)

        self.centre_label = self.text_canvas.create_text(self.initial_width / 2, self.initial_height / 2, anchor="c",
                                                         font="Helvetica", text=label_text)


