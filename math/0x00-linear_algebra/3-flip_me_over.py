#!/usr/bin/env python3
"flip_me_over"


def matrix_transpose(matrix):
    "returns the transpose of a 2D matrix"
    result = []
    for i in range(len(matrix[0])):
        final = []
        for j in range(len(matrix)):
            final.append(matrix[j][i])
        result.append(final)
    return result
