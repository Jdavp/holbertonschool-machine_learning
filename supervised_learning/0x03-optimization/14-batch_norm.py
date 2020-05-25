#!/usr/bin/env python3
'Batch Normalization Upgraded'
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    'batch normalization layer for a neural network in tensorflow'
    k_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    output = tf.layers.Dense(units=n, kernel_initializer=k_init)
    Z = output(prev)

    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[n]),
                        name="gamma")
    beta = tf.Variable(initial_value=tf.constant(0.0, shape=[n]), name="beta")
    mean, var = tf.nn.moments(Z, axes=0)
    norm_batch = tf.nn.batch_normalization(Z, mean, var, offset=beta,
                                           scale=gamma,
                                           variance_epsilon=1e-8)
    if activation:
        return activation(norm_batch)
    else:
        return norm_batch
