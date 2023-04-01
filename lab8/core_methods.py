import numpy as np
import math
from itertools import combinations


def calculate_hci_and_weights(matrix):
    n = np.shape(matrix)[0]
    an_columns_sums = matrix.sum(axis=0)
    an_weights = an_columns_sums ** -1
    hm = n / np.sum(an_weights)
    hci = ((hm - n) * (n + 1)) / (n * (n - 1))
    return hci, an_weights


def calculate_weights_by_distributive_synthesis(normalized_local_weights, criteria_weights):
    global_weights = []
    m = normalized_local_weights.shape[1]
    n = normalized_local_weights.shape[0]
    for i in range(n):
        weight = 0
        for j in range(m):
            r = normalized_local_weights[i][j] / (normalized_local_weights.sum(axis=0)[j])
            weight += (criteria_weights[j] * r)
        global_weights.append(weight)
    return global_weights


def calculate_weights_by_ideal_synthesis(normalized_local_weights, criteria_weights):
    global_weights = []
    m = normalized_local_weights.shape[1]
    n = normalized_local_weights.shape[0]
    for i in range(n):
        weight = 0
        for j in range(m):
            r = normalized_local_weights[i][j] / (np.max(normalized_local_weights[j]))
            weight += (criteria_weights[j] * r)
        global_weights.append(weight)
    global_weights = normalize_weights(np.array(global_weights))
    return global_weights


def calculate_weights_by_multiplicative_synthesis(normalized_local_weights, criteria_weights):
    global_weights = []
    m = normalized_local_weights.shape[1]
    n = normalized_local_weights.shape[0]
    for i in range(n):
        weight = 1
        for j in range(m):
            weight *= math.pow(normalized_local_weights[i][j], criteria_weights[j])
        global_weights.append(weight)
    global_weights = normalize_weights(np.array(global_weights))
    return global_weights


def calculate_weights_by_gabrpa_synthesis(full_criteria_matrix, criteria_weights):
    full_comparisons_matrix = []
    for matrix in full_criteria_matrix:
        n = matrix.shape[0]
        indexes_mask = np.linspace(0, n - 1, n, dtype=int)
        indexes_mask = np.array(list(combinations(indexes_mask, 2)))
        pairs_to_compare = np.array(list(combinations(matrix, 2)))
        criteria_comparisons = np.ones_like(matrix)
        for i in range(pairs_to_compare.shape[0]):
            weight = 1
            for j in range(pairs_to_compare.shape[2]):
                weight *= ((pairs_to_compare[i][0][j] / pairs_to_compare[i][1][j]) ** criteria_weights[0])
            index_pair = indexes_mask[i]
            criteria_comparisons[index_pair[0]][index_pair[1]] = weight
            criteria_comparisons[index_pair[1]][index_pair[0]] = weight ** -1
        full_comparisons_matrix.append(criteria_comparisons)
    full_comparisons_matrix = np.array(full_comparisons_matrix)
    full_comparisons_normalized_weights = []
    for matrix in full_comparisons_matrix:
        _, weights = calculate_hci_and_weights(matrix)
        full_comparisons_normalized_weights.append(weights)
    full_comparisons_normalized_weights = np.transpose(np.array(full_comparisons_normalized_weights))
    global_comparisons_weights = calculate_weights_by_distributive_synthesis(full_comparisons_normalized_weights,
                                                                             criteria_weights)
    return global_comparisons_weights


def normalize_weights(weights):
    return weights / np.sum(weights)
