#!/usr/bin/env python3
'Save and Load Model'
import tensorflow.keras as K


def save_model(network, filename):
    'saves an entire model'
    K.models.save_model(network, filename)
    return None


def load_model(filename):
    'loads an entire model'
    model = K.models.load_model(filename)
    return model
