#!/usr/bin/env python3
import tensorflow as tf
'Placeholders'


def create_placeholders(nx, classes):
    'returns two placeholders, x and y, for the neural network'
    x = tf.placeholder("float32", shape=[None, nx], name='x')
    y = tf.placeholder("float32", [None, classes], name='y')
    return x, y
