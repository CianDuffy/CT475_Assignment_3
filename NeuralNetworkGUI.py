import Tkinter
from tkFileDialog import askopenfilename
import ntpath
import numpy as np
from NeuralNetworkDriver import NeuralNetworkDriver


class NeuralNetworkGUI(object):
    def __init__(self):
        """
        GUI Class for the Neural Network Classifier
        Allows the user to import datasets, classify them with the Neural Network Classifier and view the results
        """
        self.main_window = Tkinter.Tk()
        self.main_window.wm_title("Artificial Neural Network")
        self.main_window.resizable(width=False, height=False)
        self.initial_width = 700
        self.initial_height = 550
        self.main_window.geometry('{}x{}'.format(self.initial_width, self.initial_height))

        self.top_label_text = Tkinter.StringVar()
        self.top_label = Tkinter.Label(self.main_window, textvariable=self.top_label_text, font=("Helvetica", 24))
        self.top_label_text.set("Import your dataset to get started")
        self.top_label.pack(pady=10, padx=20)

        self.import_button = Tkinter.Button(self.main_window, text="Import Dataset...",
                                            command=self.on_click_import_button)
        self.import_button.pack()

        self.classification_details_container = Tkinter.Frame()
        self.classification_details_label = Tkinter.Label()
        self.filename = ""
        self.filepath = ""
        self.final_results = []
        self.hidden_layer_sizes = range(10, 51, 10)
        self.training_iterations = range(1000, 10001, 1000)

        self.main_window.mainloop()

    def on_click_import_button(self):
        """
        Action for when import button is pressed
        :return: None
        """
        opts = {'filetypes': [('CSV files', '.csv')]}
        self.filepath = askopenfilename(**opts)  # show an "Open" dialog box and return the path to the selected file
        self.filename = ntpath.basename(self.filepath)
        if self.filename != "":

            self.top_label_text.set(self.filename + " successfully imported")

            self.import_button.destroy()
            float_width = float(self.initial_width)
            float_height = float(self.initial_height)

            bottom_buttons_y_position = (float_height - 20.0) / float_height
            reset_button_x_position = 20.0 / float_width
            classify_button_x_position = (float_width - 20.0) / float_width

            self.reset_button = Tkinter.Button(self.main_window, text="Reset", command=self.on_click_reset_button)
            self.reset_button.place(relx=reset_button_x_position, rely=bottom_buttons_y_position, anchor="sw")

            self.classify_button = Tkinter.Button(self.main_window, text=("Classify " + self.filename),
                                                  command=self.on_click_classify_button)
            self.classify_button.place(relx=classify_button_x_position, rely=bottom_buttons_y_position, anchor="se")

    def on_click_classify_button(self):
        """
        Action for when classify button is pressed
        :return: None
        """

        self.reset_button.configure(state="disabled")
        self.classify_button.configure(state="disabled")
        self.top_label_text.set("Classifying " + self.filename + "...")

        self.main_window.after(500, self.begin_classification)

    def on_click_reset_button(self):
        """
        Action for when reset button is pressed
        :return: None
        """

        self.reset_button.destroy()
        self.classify_button.destroy()
        self.classification_details_label.destroy()
        self.classification_details_container.destroy()

        self.filename = ""
        self.filepath = ""

        self.import_button = Tkinter.Button(self.main_window, text="Import Dataset...",
                                            command=self.on_click_import_button)
        self.import_button.pack()

        self.top_label_text.set("Import your dataset to get started")

    @staticmethod
    def matched_result_list_for_input_lists(target, result):
        matched_list = []
        for index in range(0, len(target), 1):
            if target[index] == result[index]:
                matched_list.append("Correct")
            else:
                matched_list.append("Incorrect")

        return np.array(matched_list)

    def begin_classification(self):
        """
        Passes a reference to the dataset to a NeuralNetworkDriver and displays the results
        :return: None
        """

        # Initialise NeuralNetworkDriver and have it build, train and test the Network
        driver = NeuralNetworkDriver(self.filepath)
        self.final_results = driver.build_network_and_classify_data()

        # Display the classification results in the GUI window
        self.display_classification_results()

        accuracy_sum = 0
        for result_dictionary in self.final_results:
            accuracy_sum += float(result_dictionary['accuracy'])

        # Calculate the mean accuracy and display it for the user
        mean_accuracy = accuracy_sum / len(self.final_results)
        self.reset_button.configure(state="normal")
        self.classify_button.configure(state="normal")
        self.top_label_text.set("Classification complete!\nMean accuracy: " + str(round(100 * mean_accuracy, 2)) + "%")
        self.main_window.mainloop()

    def display_classification_results(self):
        """
        Displays the results from the classification in a Tkinter text area
        :return: None
        """
        self.classification_details_label = Tkinter.Label(self.main_window, text="Classification Details:",
                                                          font=("Helvetica", 16))
        self.classification_details_label.pack(anchor="w", pady=10, padx=20)

        self.classification_details_container = Tkinter.Frame(self.main_window, bd=1, relief="solid")

        scrollbar = Tkinter.Scrollbar(self.classification_details_container)
        scrollbar.pack(side="right", fill="y")

        results_text_area = Tkinter.Text(self.classification_details_container, wrap="word", font="Helvetica")
        results_text_area.pack(fill="x")

        results_text_area.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=results_text_area.yview)
        index = 1

        for results_dictionary in self.final_results:
            input_array = results_dictionary['input']
            result_array = results_dictionary['result']
            match_array = self.matched_result_list_for_input_lists(input_array, result_array)
            accuracy = float(results_dictionary['accuracy'])
            results_text_area.insert("end", "------------------------------------- Iteration #" + str(index)
                                     + " -------------------------------------\n")
            results_text_area.insert("end", "\nTarget Classes:\n")
            results_text_area.insert("end", self.inline_string_for_array(input_array) + "\n")
            results_text_area.insert("end", "\nPredicted Classes:\n")
            results_text_area.insert("end", self.inline_string_for_array(result_array) + "\n")
            results_text_area.insert("end", "\nCorrect Predictions:\n")
            results_text_area.insert("end", self.inline_string_for_array(match_array) + "\n")
            results_text_area.insert("end", "\nPrediction Accuracy: " + str(round(100 * accuracy, 2)) + "%\n\n")
            index += 1

        results_text_area.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=results_text_area.yview)

        self.classification_details_container.pack(fill="x", padx=20)

    @staticmethod
    def inline_string_for_array(input_array):
        """
        Converts an array into an inline string of values separated by commas
        :param input_array: Array of items to be added to string
        :return: Inline string of values separated by commas
        """
        inline_string = ""
        index = 1
        for entry in input_array:
            inline_string += entry
            # Add a comma for all but the last entry
            if index < len(input_array):
                inline_string += ", "
            index += 1
        return inline_string
