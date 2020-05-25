'Put it all together and what do you get?'
import tensorflow as tf
import numpy as np


def create_placeholders(nx, classes):
    'returns two placeholders, x and y, for the neural network'
    x = tf.placeholder("float32", shape=[None, nx], name='x')
    y = tf.placeholder("float32", [None, classes], name='y')
    return x, y


def create_layer(prev, n, activation):
    """prev is the tensor output of the previous layer
    n is the number of nodes in the layer to create
    activation is the activation function that the layer should use
    use tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    to implement He et. al initialization for the layer weights
    each layer should be given the name layer
    Returns: the tensor output of the layer"""

    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            mode='FAN_AVG'),
        name='layer',
    )

    return layer(prev)


def forward_prop(x, layer_sizes=[], activations=[]):
    'creates the forward propagation graph for the neural network'
    z = create_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        z = create_layer(z, layer_sizes[i], activations[i])
    return z


def calculate_accuracy(y, y_pred):
    'calculates the accuracy of a prediction'
    predict = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    num = tf.cast(predict, tf.float32)
    mean = tf.reduce_mean(num)
    return mean


def calculate_loss(y, y_pred):
    'calculates the softmax cross-entropy loss of a prediction'
    return tf.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=y_pred,
        reduction=tf.losses.Reduction.MEAN
        )


def shuffle_data(X, Y):
    ' shuffles the data points in two matrices the same way'
    perm = np.random.permutation(len(X))
    return X[perm], Y[perm]


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    'tensorflow using the Adam optimization algorithm'
    return tf.train.AdamOptimizer(
        learning_rate=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
    ).minimize(loss=loss)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    'tensorflow using inverse time decay'

    return tf.train.inverse_time_decay(
        learning_rate=alpha,
        global_step=global_step,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True,
    )


def create_batch_norm_layer(prev, n, activation):
    'batch normalization layer for a neural network in tensorflow'
    k_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    output = tf.layers.Dense(units=n, kernel_initializer=k_init)
    Z = output(prev)

    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[n]),
                        name="gamma")
    beta = tf.Variable(initial_value=tf.constant(0.0, shape=[n]), name="beta")
    mean, var = tf.nn.moments(Z, axes=0)
    norm_batch = tf.nn.batch_normalization(Z, mean, var, offset=beta,
                                           scale=gamma,
                                           variance_epsilon=1e-8)
    if activation:
        return activation(norm_batch)
    else:
        return norm_batch


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32,
          epochs=5, save_path='/tmp/model.ckpt'):
    'trains a loaded neural network model using mini-batch gradient descent'
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path+'.meta')
        saver.restore(sess, load_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        # mini_batches#
        data = X_train.shape[0]

        if data % batch_size == 0:
            m_batch = data//batch_size
        else:
            m_batch = data//batch_size+1
        # epochs
        for epoca in range(epochs+1):
            train_dict = {x: X_train, y: Y_train}
            valid_dict = {x: X_valid, y: Y_valid}
            # train
            train_cost = sess.run(loss, feed_dict=train_dict)
            train_accuracy = sess.run(accuracy, feed_dict=train_dict)
            # valid
            valid_cost = sess.run(loss, feed_dict=valid_dict)
            valid_accuracy = sess.run(accuracy, feed_dict=valid_dict)

            print("After {} epochs:".format(epoca))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoca < epochs:
                shuffle_x, shuffle_y = shuffle_data(X_train, Y_train)

                for batch in range(m_batch):
                    start = batch*batch_size
                    end = (batch+1)*batch_size
                    if end > data:
                        end = data
                    m_batch_X = shuffle_x[start:end]
                    m_batch_Y = shuffle_y[start:end]

                    new_train = {x: m_batch_X, y: m_batch_Y}
                    sess.run(train_op, feed_dict=new_train)

                    if (batch+1) % 100 == 0:
                        batch_cost = sess.run(loss, feed_dict=new_train)
                        batch_accuracy = sess.run(accuracy,
                                                  feed_dict=new_train)
                        print("\tStep {}:".format(batch+1))
                        print("\t\tCost: {}".format(batch_cost))
                        print("\t\tAccuracy: {}".format(batch_accuracy))
        return saver.save(sess, save_path)
