#!/usr/bin/env python3
"Across The Planes"


def matrix_shape(matrix):
    "returns the total size of all the list in a matrix"
    total = []
    item = matrix
    while type(item) is list:
        total.append(len(item))
        item = item[0]
        return total


def add_matrices2D(mat1, mat2):
    "adds two matrices element-wise"
    if matrix_shape(mat1) != matrix_shape(mat2[0]):
        return None
    else:
        return [[mat1[x][y] + mat2[x][y] for y in range(len(mat1[0]))]
                for x in range(len(mat1))]
