import numpy as np


def sub_cost(x, y, mode=2):
    if x == y:
        return 0
    else:
        return mode


def edit_distance(source, target, del_cost=1, ins_cost=1):
    n = len(source)
    m = len(target)
    distance_matrix = np.zeros((n+1, m+1))
    for row in range(1, n+1):
        distance_matrix[row, 0] = distance_matrix[row-1, 0] + del_cost
    for column in range(1, m+1):
        distance_matrix[0, column] = distance_matrix[0, column-1] + ins_cost
    
    for row in range(1, n+1):
        for column in range(1, m+1):
            distance_matrix[row, column] = min(
                distance_matrix[row-1, column] + del_cost,
                distance_matrix[row-1, column-1] +
                sub_cost(source[row-1], target[column-1]),
                distance_matrix[row, column-1] + ins_cost)

    return distance_matrix


print(edit_distance("intention", "execution"))

    