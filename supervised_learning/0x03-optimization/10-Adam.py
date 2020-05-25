#!/usr/bin/env python3
"Adam Upgraded"
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    'tensorflow using the Adam optimization algorithm'
    return tf.train.AdamOptimizer(
        learning_rate=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
    ).minimize(loss=loss)
