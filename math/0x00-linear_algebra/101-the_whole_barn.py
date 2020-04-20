#!/usr/bin/env python3
'The Whole Barn'


def add_matrices(mat1, mat2):
    'adds two matrices'
    if type(mat1[0]) is list or type(mat2[0]) is list:
        if len(mat1) != len(mat2[0]):
            return None
        if type(mat1[0]) is list:
            result = [[mat1[i][j] + mat2[i][j] for j in range
                       (len(mat1[0]))] for i in range(len(mat1))]
    else:
        result = [a+b for a, b in zip(mat1, mat2)]
    return result
