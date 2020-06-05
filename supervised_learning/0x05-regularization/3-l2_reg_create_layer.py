#!/usr/bin/env python3
'Create a Layer with L2 Regularization'
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    'tensorflow layer that includes L2 regularization'
    l2 = tf.contrib.layers.l2_regularizer(lambtha)
    k_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=k_init,
                            kernel_regularizer=l2)
    return layer(prev)
