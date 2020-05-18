#!/usr/bin/env python3
'Calculates Train_Op '
import tensorflow as tf


def create_train_op(loss, alpha):
    'training operation for the network'
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(loss)

    return train
