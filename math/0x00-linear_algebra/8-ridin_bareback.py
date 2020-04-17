#!/usr/bin/env python3
"Across The Planes"


def mat_mul(mat1, mat2):
    "adds two matrices element-wise"
    if len(mat1[0]) != len(mat2):
        return None
    else:
        return [[sum(x*y for x, y in zip(mat1_row, mat2_col))
                 for mat2_col in zip(*mat2)] for mat1_row in mat1]
