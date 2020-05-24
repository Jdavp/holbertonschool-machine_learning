#!/usr/bin/env python3
'Mini-Batch'
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
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
