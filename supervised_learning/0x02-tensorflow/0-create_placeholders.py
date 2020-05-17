#!/usr/bin/env python3
'Placeholders'
import tensorflow as tf


def create_placeholders(nx, classes):
    'returns two placeholders, x and y, for the neural network'
    x = tf.placeholder("float32", shape=[None, nx], name='x')
    y = tf.placeholder("float32", [None, classes], name='y')
    return x, y
