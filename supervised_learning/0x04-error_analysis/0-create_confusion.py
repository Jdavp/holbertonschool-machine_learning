#!/usr/bin/env python3
'Create Confusion'
import numpy as np


def create_confusion_matrix(labels, logits):
    'creates a confusion matrix'
    return labels.T.dot(logits)
