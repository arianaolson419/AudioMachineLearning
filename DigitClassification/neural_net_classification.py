"""Perform classification on handwritten digits from the MNIST data set (as part of the
Kaggle Digit Recognizer challenge).

Author: Ariana Olson
"""

import numpy as np
import pandas as pd
import tensorflow as tf

CSV_COLUMN_NAMES_TEST = ['pixel{}'.format(x) for x in range(784)]
CSV_COLUMN_NAMES_TRAIN = list(CSV_COLUMN_NAMES_TEST)
CSV_COLUMN_NAMES_TRAIN.insert(0, 'label')

def load_data(label_name='label'):
    # Parse the csv file
    train = pd.read_csv('data/train.csv', names=CSV_COLUMN_NAMES_TRAIN, header=0)
    
    train_features, train_label = train, train.pop(label_name)

    test = pd.read_csv('data/test.csv', names=CSV_COLUMN_NAMES_TEST, header=0)
    test_features = test

    return (train_features, train_label), (test_features)

def cnn_model_fn_train(features, labels, mode):
    """Model function for CNN."""
    # Input layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional layer #1
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2d, pool_size=[2, 2], strides=2)
    
    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
            # Generate predictions (for PREDICT mode)
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # Calculate Loss (for TRAIN mode)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

def main(unused_argv):
    pass


if __name__ == "__main__":
    load_data()
