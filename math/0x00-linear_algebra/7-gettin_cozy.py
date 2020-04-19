#!/usr/bin/env python3
"Gettin Cozy"


def deepcopy(obj):
    if isinstance(obj, dict):
        return {deepcopy(key): deepcopy(value) for key, value in obj.items()}
    if hasattr(obj, '__iter__'):
        return type(obj)(deepcopy(item) for item in obj)
    return obj


def cat_matrices2D(mat1, mat2, axis=0):
    "concatenates two matrices along a specific axis"
    copy_mat1 = deepcopy(mat1)
    copy_mat2 = deepcopy(mat2)
    if (axis == 1):
        return [x + y for x, y in zip(copy_mat1, copy_mat2)]
    else:
        new_concat = copy_mat1 + copy_mat2
        return new_concat
