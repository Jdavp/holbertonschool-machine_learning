#!/usr/bin/env python3
'Accuracy '
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    'calculates the accuracy of a prediction'
    predict = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    num = tf.cast(predict, tf.float32)
    mean = tf.reduce_mean(num)
    return mean
