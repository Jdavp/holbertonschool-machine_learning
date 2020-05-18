#!/usr/bin/env python3
'Calculate Loss'
import tensorflow as tf


def calculate_loss(y, y_pred):
    'calculates the softmax cross-entropy loss of a prediction'
    return tf.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=y_pred,
        reduction=tf.losses.Reduction.MEAN
        )
