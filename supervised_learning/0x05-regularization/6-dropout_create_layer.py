#!/usr/bin/env python3
'Create a Layer with Dropout'
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    'creates a layer of a neural network using dropout'
    dropout = tf.layers.Dropout(keep_prob)
    k_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=k_init,
                            kernel_regularizer=dropout)
    return layer(prev)
