#!/usr/bin/env python3
'Sensitivity'
import numpy as np


def sensitivity(confusion):
    'calculates the sensitivity for each class in a confusion matrix'
    TP = np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    return TP/(TP+FN)
