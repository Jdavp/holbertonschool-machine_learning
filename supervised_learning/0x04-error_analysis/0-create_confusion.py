#!/usr/bin/env python3
'Create Confusion'


def create_confusion_matrix(labels, logits):
    'creates a confusion matrix'
    return labels.T.dot(logits)
