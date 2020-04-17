#!/usr/bin/env python3
"Across The Planes"


def add_matrices2D(mat1, mat2):
    "adds two matrices element-wise"
    return [[mat1[x][y] + mat2[x][y] for y in range(len(mat1[0]))]
            for x in range(len(mat1))]
