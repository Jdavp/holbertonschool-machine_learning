#!/usr/bin/env python3
'Early Stopping'
import tensorflow.keras as K


def train_model(network, data,
                labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    'train the model using early stopping'
    if validation_data and early_stopping:
        callbacks = [
                    K.callbacks.EarlyStopping(patience=patience, monitor='val_loss'),
                ]
        return network.fit(data, labels, batch_size=batch_size,
                           epochs=epochs, verbose=verbose,
                           shuffle=shuffle, validation_data=validation_data,
                           callbacks=callbacks)
