#!/usr/bin/env python3
"""matrix size"""
def matrix_shape(matrix):
    "returns the total size of all the list in a matrix"
    total = []
    item = matrix
    while type(item) is list:
        total.append(len(item))
        item = item[0]
    return total
