#!/usr/bin/env python3
"Gettin Cozy"
import copy


def cat_matrices2D(mat1, mat2, axis=0):
    "concatenates two matrices along a specific axis"
    copy_mat1 = copy.deepcopy(mat1)
    copy_mat2 = copy.deepcopy(mat2)
    if (axis == 1):
        return [x + y for x, y in zip(copy_mat1, copy_mat2)]
    else:
        new_concat = copy_mat1 + copy_mat2
        return new_concat
