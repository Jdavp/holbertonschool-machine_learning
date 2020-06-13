#!/usr/bin/env python3
'LeNet-5 (Keras)'
import tensorflow.keras as K


def lenet5(X):
    'a keras modified version of the LeNet-5 architecture'
    init = K.initializers.he_normal()
    activation = "relu"

    conv1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5), padding="same",
                            activation=activation, kernel_initializer=init)(X)
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = K.layers.Conv2D(filters=16, kernel_size=(5, 5), padding="valid",
                            activation=activation, kernel_initializer=init)(
        pool1)
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    flatten = K.layers.Flatten()(pool2)

    fc1 = K.layers.Dense(units=120, activation=activation,
                         kernel_initializer=init)(flatten)

    fc2 = K.layers.Dense(units=84, activation=activation,
                         kernel_initializer=init)(fc1)

    fc3 = K.layers.Dense(units=10, activation="softmax",
                         kernel_initializer=init)(fc2)

    model = K.models.Model(inputs=X, outputs=fc3)

    opt = K.optimizers.Adam()

    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])

    return model
