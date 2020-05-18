#!/usr/bin/env python3
'Accuracy '
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    'calculates the accuracy of a prediction'
    predict = tf.equal(y, y_pred)
    num = tf.cast(predict, tf.float32)
    mean = tf.reduce_mean(num)
    return mean
