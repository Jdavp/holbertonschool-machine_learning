#!/usr/bin/env python3
'Train'
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    'trains a model using mini-batch gradient descent'
    return network.fit(data, labels, epochs=epochs,
                       batch_size=batch_size, verbose=verbose,
                       shuffle=shuffle)
