#!/usr/bin/env python3
'Train'
import tensorflow.keras as K


def train_model(network, data,
                labels, batch_size,
                epochs, validation_data=None,
                verbose=True, shuffle=False):
    'also analyze validation data'
    if validation_data is not None:
        return network.fit(data, labels, batch_size=batch_size,
                           epochs=epochs, verbose=verbose,
                           shuffle=shuffle, validation_data=validation_data)
