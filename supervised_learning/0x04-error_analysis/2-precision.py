#!/usr/bin/env python3
'Precision'
import numpy as np


def precision(confusion):
    'calculates the precision for each class in a confusion matrix'
    TP = np.diag(confusion)
    FP = confusion.sum(axis=0) - np.diag(confusion)
    return TP/(TP+FP)
