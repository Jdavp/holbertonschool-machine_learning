#!/usr/bin/env python3
'Optimize'
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    'Adam with categorical crossentropy loss and accuracy metrics'
    adam = tf.keras.optimizers.Adam(
        beta_1=beta1,
        beta_2=beta2,
        decay=alpha
    )
    network.compile(loss="binary_crossentropy", optimizer=adam)
    return None
