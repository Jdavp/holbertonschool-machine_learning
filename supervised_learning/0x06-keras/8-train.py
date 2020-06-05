#!/usr/bin/env python3
'Save Only the Best'
import tensorflow.keras as K


def train_model(network, data,
                labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    'save the best iteration of the model'
    def scheduler(epoch):
        '''Function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            learning rate as output '''
        return alpha / (1 + decay_rate * epoch)

    callbacks = []
    if early_stopping and validation_data:
        callbacks.append(K.callbacks.EarlyStopping(patience=patience))
    if learning_rate_decay and validation_data:
        callbacks.append(K.callbacks.LearningRateScheduler(
                         scheduler, verbose=1))
    if save_best and filepath:
        callbacks.append(K.callbacks.ModelCheckpoint(filepath=filepath,
                         save_best_only=True, monitor='val_loss', mode='min'))

    return network.fit(data, labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose,
                       shuffle=shuffle, validation_data=validation_data,
                       callbacks=callbacks)
