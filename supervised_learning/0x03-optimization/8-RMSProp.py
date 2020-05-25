#!/usr/bin/env python3
'RMSProp Upgraded'
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    'Tensorflow using the RMSProp optimization algorithm'

    return tf.train.RMSPropOptimizer(
        learning_rate=alpha,
        decay=beta2,
        epsilon=epsilon
        ).minimize(loss)
