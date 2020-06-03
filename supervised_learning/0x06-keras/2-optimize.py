#!/usr/bin/env python3
'Optimize'
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    'Adam with categorical crossentropy loss and accuracy metrics'
    adam = K.optimizers.Adam(
        alpha,
        beta_1=beta1,
        beta_2=beta2,
    )
    network.compile(loss="categorical_crossentropy", optimizer=adam,
                    metrics=["accuracy"])
    return None
