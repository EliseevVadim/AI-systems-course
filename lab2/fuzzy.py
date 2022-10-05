import numpy as np
import math
from matplotlib import pyplot as plt
from shapely.geometry import Polygon


class FuzzyVariable(object):

    def __init__(self, low_elements_bounds, middle_elements_bounds,
                 high_elements_bounds):
        self.low_values = np.arange(low_elements_bounds[0], low_elements_bounds[1] + 0.1, 0.1)
        self.middle_values = np.arange(middle_elements_bounds[0], middle_elements_bounds[1] + 0.1, 0.1)
        self.high_values = np.arange(high_elements_bounds[0], high_elements_bounds[1] + 0.1, 0.1)
        self.low_possibilities = np.ones(len(self.low_values))
        self.middle_possibilities = np.ones(len(self.middle_values))
        self.high_possibilities = np.ones(len(self.high_values))
        length = len(self.low_possibilities)
        self.low_possibilities[int(length / 2): length + 1] = np.linspace(1.0, 0.0, int(length / 2) + 1)
        length = len(self.middle_possibilities)
        self.middle_possibilities[: int(length / 2) + 1] = np.linspace(0.0, 1.0, int(length / 2) + 1)
        self.middle_possibilities[int(length / 2): length + 1] = np.linspace(1.0, 0.0, int(length / 2) + 1)
        self.high_possibilities[: int(length / 2) + 1] = np.linspace(0.0, 1.0, int(length / 2) + 1)

    def plot_low_elements_graph(self):
        if self.low_possibilities.any():
            plt.plot(self.low_values, self.low_possibilities)
            plt.grid()

    def plot_middle_elements_graph(self):
        if self.middle_possibilities.any():
            plt.plot(self.middle_values, self.middle_possibilities)
            plt.grid()

    def plot_high_elements_graph(self):
        if self.high_possibilities.any():
            plt.plot(self.high_values, self.high_possibilities)
            plt.grid()

    def plot_all_graphs(self):
        self.plot_low_elements_graph()
        self.plot_middle_elements_graph()
        self.plot_high_elements_graph()


class FuzzyCalculator(object):

    def __init__(self, antecedent_variable, consequent_variable):
        self.antecedent_variable = antecedent_variable
        self.consequent_variable = consequent_variable

    def calculate(self, antecedent_value):
        try:
            value_is_low = self.__get_possibility(antecedent_value, self.antecedent_variable.low_values,
                                                  self.antecedent_variable.low_possibilities)
            value_is_middle = self.__get_possibility(antecedent_value, self.antecedent_variable.middle_values,
                                                     self.antecedent_variable.middle_possibilities)
            value_is_high = self.__get_possibility(antecedent_value, self.antecedent_variable.high_values,
                                                   self.antecedent_variable.high_possibilities)
            self.consequent_variable.low_possibilities *= value_is_low
            self.consequent_variable.middle_possibilities *= value_is_middle
            self.consequent_variable.high_possibilities *= value_is_high
            if value_is_low == 1:
                scale = len(self.consequent_variable.low_values) / len(self.antecedent_variable.low_values)
                index = np.where(np.isclose(self.antecedent_variable.low_values, antecedent_value))[0][0]
                return round(self.consequent_variable.low_values[math.ceil(index * scale)], 2)
            if value_is_middle == 1:
                scale = len(self.consequent_variable.middle_values) / len(self.antecedent_variable.middle_values)
                index = np.where(np.isclose(self.antecedent_variable.middle_values, antecedent_value))[0][0]
                return round(self.consequent_variable.middle_values[math.ceil(index * scale)], 2)
            if value_is_high == 1:
                scale = len(self.consequent_variable.high_values) / len(self.antecedent_variable.high_values)
                index = np.where(np.isclose(self.antecedent_variable.high_values, antecedent_value))[0][0]
                return round(self.consequent_variable.high_values[math.ceil(index * scale)], 2)
            if value_is_low == 0:
                index = self.__get_lines_intersection_for_right_side()
                analyzed_elements_length = len(self.consequent_variable.middle_values)
                points = [
                    [self.consequent_variable.middle_values[0], 0],
                    [self.consequent_variable.middle_values[int(analyzed_elements_length / 2)],
                     self.consequent_variable.middle_possibilities[int(analyzed_elements_length / 2)]],
                    [self.consequent_variable.middle_values[index], self.consequent_variable.middle_possibilities[index]],
                    [self.consequent_variable.high_values[int(analyzed_elements_length / 2)],
                     self.consequent_variable.high_possibilities[int(analyzed_elements_length / 2)]],
                    [self.consequent_variable.high_values[len(self.consequent_variable.high_values) - 1],
                     self.consequent_variable.high_possibilities[len(self.consequent_variable.high_possibilities) - 1]],
                    [1000, 0]
                ]
                self.__plot_decision_graph(points)
                polygon = Polygon(points)
                return round(list(polygon.centroid.coords)[0][0], 2)
            if value_is_high == 0:
                index = self.__get_lines_intersection_for_left_side()
                low_elements_length = len(self.consequent_variable.low_values)
                middle_elements_length = len(self.consequent_variable.middle_values)
                points = [
                    [0, 0],
                    [0, self.consequent_variable.low_possibilities[0]],
                    [self.consequent_variable.low_values[int(low_elements_length / 2)],
                     self.consequent_variable.low_possibilities[int(low_elements_length / 2)]],
                    [self.consequent_variable.low_values[index], self.consequent_variable.low_possibilities[index]],
                    [self.consequent_variable.middle_values[int(middle_elements_length / 2)],
                     self.consequent_variable.middle_possibilities[int(middle_elements_length / 2)]],
                    [self.consequent_variable.middle_values[middle_elements_length - 1],
                     self.consequent_variable.middle_possibilities[middle_elements_length - 1]]
                ]
                self.__plot_decision_graph(points)
                polygon = Polygon(points)
                return round(list(polygon.centroid.coords)[0][0], 2)
        except:
            print('Невозможно подсчитать, так как заданная температура лежит вне допустимого диапазона 0-60 С˚')

    def __get_possibility(self, value, values, possibilities):
        result_indexes = np.where(np.isclose(values, value))[0]
        return possibilities[result_indexes[0]] if len(result_indexes) > 0 else 0

    def __plot_decision_graph(self, points):
        self.consequent_variable.plot_all_graphs()
        for point in points:
            plt.plot(point[0], point[1], 'ro')
        plt.show()

    def plot_output_graph(self, input_data, output_data):
        plt.plot(input_data, output_data)
        for i in range(len(input_data)):
            plt.plot(input_data[i], output_data[i], 'ro')
        plt.grid()

    def __get_lines_intersection_for_right_side(self):
        actual_length = len(self.consequent_variable.middle_values)
        analyzed_length = math.ceil(len(self.consequent_variable.middle_values) / 2)
        index = np.argwhere(np.diff(np.sign(self.consequent_variable.middle_possibilities[
                                            analyzed_length - 1:actual_length + 1] - self.consequent_variable.high_possibilities[
                                                                                     :analyzed_length])).flatten())[0]
        index += analyzed_length
        return index

    def __get_lines_intersection_for_left_side(self):
        actual_length = len(self.consequent_variable.middle_values)
        analyzed_length = math.ceil(len(self.consequent_variable.middle_values) / 2)
        index = np.argwhere(np.diff(np.sign(self.consequent_variable.low_possibilities[
                                            analyzed_length - 1:actual_length + 1] - self.consequent_variable.middle_possibilities[
                                                                                     :analyzed_length])).flatten())[0]
        index += analyzed_length
        return index
