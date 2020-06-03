#!/usr/bin/env python3
'Input'
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    'builds a neural network with the Keras library'
    inputs = K.Input(shape=(nx,))
    model = K.layers.Dense(
            layers[0], activation=activations[0],
            kernel_regularizer=K.regularizers.l2(lambtha))(inputs)
    for i in range(1, len(layers)):
        model = K.layers.Dropout(1-keep_prob)(model)
        model = K.layers.Dense(
            layers[i], activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha))(model)
    model = K.Model(inputs=inputs, outputs=model)
    return model
