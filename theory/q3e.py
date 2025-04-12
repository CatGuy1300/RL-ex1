from math import inf
from typing import List, Dict, Tuple


def get_most_probable_sequence(length: int, probabilities: List[List[float]], char_mapping: Dict[str, int],
                               space_char: str) -> Tuple[str, float]:
    cost_matrix, argmax_matrix = calculate_forward_matrices(length, probabilities, char_mapping, space_char)
    best_combo = calculate_best_combination_from_argmax_matrix(length, argmax_matrix, char_mapping, space_char)
    best_prob = cost_matrix[char_mapping[space_char]][length]
    return best_combo, best_prob


def calculate_best_combination_from_argmax_matrix(length: int, argmax_matrix: List[List[str]],
                                                char_mapping: Dict[str, int], space_char: str) -> str:
    best_combo_list = ['' for _ in range(length)] + [space_char]
    for i in range(length, 0, -1):
        best_combo_list[i-1] = argmax_matrix[char_mapping[best_combo_list[i]]][i]
    return ''.join(best_combo_list[:-1])


def calculate_forward_matrices(length: int, probabilities: List[List[float]], char_mapping: Dict[str, int],
                               space_char: str) ->  Tuple[List[List[float]], List[List[str]]]:
    cost_matrix = [[-inf for _ in range(length + 1)] for _ in range(len(char_mapping))]
    argmax_matrix =  [['' for _ in range(length + 1)] for _ in range(len(char_mapping))]
    for char in char_mapping:
        cost_matrix[char_mapping[char]][0] = probabilities[char_mapping[space_char]][char_mapping[char]]

    for i in range(1, length + 1):
        for char in char_mapping:
            if (char != space_char and i != length) or (char == space_char and i == length):
                probes = [(cost_matrix[char_mapping[prev_char]][i-1] *
                           probabilities[char_mapping[prev_char]][char_mapping[char]],
                           prev_char)
                          for prev_char in char_mapping]
                max_prob, arg_max = max(probes, key=lambda tup: tup[0])
                cost_matrix[char_mapping[char]][i] = max_prob
                argmax_matrix[char_mapping[char]][i] = arg_max

    return cost_matrix, argmax_matrix


if __name__ == '__main__':
    LENGTH = 5
    PROBES = [[0.1, 0.325, 0.25, 0.325], [0.4, 0, 0.4, 0.2], [0.2, 0.2, 0.2, 0.4], [1, 0, 0, 0]]
    CHAR_MAPPING = {'b': 0, 'k': 1, 'o': 2, '-': 3}
    SPACE_CHAR = '-'

    combo, prob = get_most_probable_sequence(LENGTH, PROBES, CHAR_MAPPING, SPACE_CHAR)
    print(f'combo: {combo}, prob: {prob}')