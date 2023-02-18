import numpy as np
import math


class TrafficStrategyCreator:

    def __init__(self):
        self.constraints_coefficients = np.empty([0, 0])
        self.constraints_values = np.empty([0, 0])
        self.optimizing_function_coefficients = np.empty([0, 0])
        self.x = [float(0)] * len(self.optimizing_function_coefficients)
        self.optimal_value = None
        self.transform = False
        self.bays_count = None
        self.goods_count = None

    def add_bays_info(self, bays_volumes, bays_mass_capacities):
        if not len(bays_volumes) == len(bays_mass_capacities):
            raise ValueError("Параметры должны иметь одинаковую длину")
        self.constraints_values = np.empty([0, 0])
        self.constraints_values = np.append(self.constraints_values, bays_volumes)
        self.constraints_values = np.append(self.constraints_values, bays_mass_capacities)
        self.bays_count = len(bays_volumes)

    def add_goods_info(self, amounts_of_goods, goods_prices, goods_masses, goods_volumes):
        if not len(amounts_of_goods) == len(goods_prices):
            raise ValueError("Параметры должны иметь одинаковую длину")
        self.constraints_values = np.append(self.constraints_values, amounts_of_goods)
        self.optimizing_function_coefficients = np.array([[element] * 3 for element in goods_prices]).ravel()
        self.transform = False
        self.goods_count = len(amounts_of_goods)
        self.constraints_coefficients = np.empty([0, self.bays_count * self.goods_count])
        self.add_info_array_to_constraint_coefficients(goods_masses)
        self.add_info_array_to_constraint_coefficients(goods_volumes)
        self.apply_greater_than_zero_constraint()

    def add_info_array_to_constraint_coefficients(self, info_array):
        for i in range(self.bays_count):
            appending_array = [[0] * (self.bays_count * self.goods_count)]
            position = i
            for info in info_array:
                appending_array[0][position] = info
                position += self.bays_count
            self.constraints_coefficients = np.concatenate((self.constraints_coefficients, appending_array))

    def add_constraints_coefficients(self, constraints_coefficients):
        self.constraints_coefficients = constraints_coefficients

    def apply_greater_than_zero_constraint(self):
        for i in range(self.goods_count):
            appending_array = [[0] * (self.bays_count * self.goods_count)]
            for j in range(i * self.bays_count, self.bays_count * (i + 1)):
                appending_array[0][j] = 1
            self.constraints_coefficients = np.concatenate((self.constraints_coefficients, appending_array))

    def print_solution(self):
        print("Полученные коэффициенты:")
        print(self.x)
        print()
        print("Оптимальная выручка:")
        print(round(self.optimal_value, 1))
        print()
        print("Для получения оптимальной выручки необходимо:")
        res = [sum(self.x[i: i + self.bays_count]) for i in range(0, len(self.x), self.bays_count)]
        for i in range(len(res)):
            print(f"{round(res[i])} ед. товара {i + 1}")

    def print_simplex_table(self, simplex_table):
        print("Базис\t", end="")
        print("B\t", end="")
        for j in range(0, len(self.optimizing_function_coefficients)):
            print("x_" + str(j), end="\t")
        for j in range(0, (len(simplex_table[0]) - len(self.optimizing_function_coefficients) - 2)):
            print("y_" + str(j), end="\t")
        print()
        for j in range(0, len(simplex_table)):
            for i in range(0, len(simplex_table[0])):
                if not np.isnan(simplex_table[j, i]):
                    if i == 0:
                        print(int(simplex_table[j, i]), end="\t")
                    else:
                        print(round(simplex_table[j, i], 2), end="\t")
                else:
                    print(end="\t")
            print()

    def construct_simplex_table(self):
        number_of_counting_variables = len(self.optimizing_function_coefficients)
        number_of_slack_variables = len(self.constraints_coefficients)
        t1 = np.hstack(([None], [0], self.optimizing_function_coefficients, [0] * number_of_slack_variables))
        basis = np.array([0] * number_of_slack_variables)
        for i in range(0, len(basis)):
            basis[i] = number_of_counting_variables + i
        constraints_coefficients = self.constraints_coefficients
        if not ((number_of_slack_variables + number_of_counting_variables) == len(self.constraints_coefficients[0])):
            constraints_values = np.identity(number_of_slack_variables)
            constraints_coefficients = np.hstack((self.constraints_coefficients, constraints_values))
        t2 = np.hstack((np.transpose([basis]), np.transpose([self.constraints_values]), constraints_coefficients))
        table = np.vstack((t1, t2))
        table = np.array(table, dtype='float')
        return table

    def optimize(self):
        simplex_table = self.construct_simplex_table()
        print("Исходная таблица:")
        self.print_simplex_table(simplex_table)

        # начальный базис не оптимален
        optimal = False
        iteration = 1
        while True:
            print("----------------------------------")
            print("Итерация :", iteration)
            self.print_simplex_table(simplex_table)
            for profit in simplex_table[0, 2:]:
                if profit > 0:
                    optimal = False
                    break
                optimal = True
            if optimal:
                break

            # поиск ведущего столбца
            pivot_column = simplex_table[0, 2:].tolist().index(np.amax(simplex_table[0, 2:])) + 2

            # поиск ведущей строки
            minimum = math.inf
            pivot_row = -1

            for i in range(1, len(simplex_table)):
                if simplex_table[i, pivot_column] > 0:
                    val = simplex_table[i, 1] / simplex_table[i, pivot_column]
                    if val < minimum:
                        minimum = val
                        pivot_row = i

            pivot = simplex_table[pivot_row, pivot_column]

            print("Ведущий столбец:", pivot_column)
            print("Ведущая строка:", pivot_row)
            print("Ведущий элемент: ", pivot)

            # деление ведущей строки на ведущий элемент
            simplex_table[pivot_row, 1:] = simplex_table[pivot_row, 1:] / pivot

            # обновление таблицы с учетом ведущего элемента
            for i in range(0, len(simplex_table)):
                if i != pivot_row:
                    multiplier = simplex_table[i, pivot_column] / simplex_table[pivot_row, pivot_column]
                    simplex_table[i, 1:] = simplex_table[i, 1:] - multiplier * simplex_table[pivot_row, 1:]

            # добавление базисной переменной
            simplex_table[pivot_row, 0] = pivot_column - 2

            iteration += 1

        print("----------------------------------")
        print("Результирующая таблица была получена за ", iteration, "интераций")
        self.print_simplex_table(simplex_table)

        self.x = np.array([0] * len(self.optimizing_function_coefficients), dtype=float)
        # сохранение коэффициентов
        for key in range(1, (len(simplex_table))):
            if simplex_table[key, 0] < len(self.optimizing_function_coefficients):
                self.x[int(simplex_table[key, 0])] = simplex_table[key, 1]

        self.optimal_value = -1 * simplex_table[0, 1]
