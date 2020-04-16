#!/usr/bin/env python3
"line up"


def add_arrays(arr1, arr2):
    "that adds two arrays element-wise"

    if len(arr1) != len(arr2):
        return None
    else:
        return[x + y for x, y in zip(arr1, arr2)]
