"""Implement a fully connected digit classification network using TensorFlow
"""

# Import dependencies
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm   # prints progress bar to the terminal
import sys

# Import formatting script
from format_data import load_train_and_validation_data

# Load numpy arrays of formatted data from the Kaggle dataset
train_data, train_labels, validate_data, validate_labels = load_train_and_validation_data(1000) 

# Set training parameters
# TODO add flags for setting parameters
batch_size = 500

if len(sys.argv) > 1:
    learning_rate = float(sys.argv[1])
else:
    learning_rate = 0.001

# Partition the data into batched for training and validating the model.
batched_train_data = np.reshape(train_data, (-1, batch_size, 784))
batched_train_labels = np.reshape(train_labels, (-1, batch_size))

batched_validate_data = np.reshape(validate_data, (-1, batch_size, 784))
batched_validate_labels = np.reshape(validate_labels, (-1, batch_size))

# This will be the initializer used for the weights and biases.
he = tf.contrib.keras.initializers.he_normal

# The input layer is a tensorflow placeholder that will be given values when
# the session is run.
input_layer = tf.placeholder(dtype=tf.float32, shape=(None, 784))

# The hidden and output layers are densely connected. Each takes the previous
# layer as input.
hidden_layer_1 = tf.layers.dense(input_layer, 30, activation=tf.nn.sigmoid,
        kernel_initializer = he(), bias_initializer = he())
output_layer = tf.layers.dense(hidden_layer_1, 10, activation=tf.nn.sigmoid,       
        kernel_initializer=he(), bias_initializer=he())

# The labels placeholder will be filled in with the label of the datapoint when
# the session is run.
labels = tf.placeholder(shape=(None), dtype=tf.int32)

# Define the loss function. The chosen loss function works with the labels
# represented as integers, the way they are in the dataset.
loss = tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output_layer, labels=labels))

# Define the optimization function.
opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# All of the training and testing must be run in the session. This is where the
# symbolic network model is given actual values.
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    # Iterate through all of the training and validation data
    train_iters = len(batched_train_data)
    test_iters = len(batched_validate_data)
    
    train_losses = []
    acc = 0

    # Train the network in batches
    for i in tqdm(range(train_iters)):
        images = batched_train_data[i]
        lbls = batched_train_labels[i]
        _, lossval = session.run((opt, loss), feed_dict = {labels: lbls, input_layer: images})
        if i % 10 == 0:
            train_losses.append(lossval)

    # Test the network in batched
    for i in tqdm(range(test_iters)):
        images = batched_validate_data[i]
        lbls = batched_validate_labels[i]
        lbl = session.run(tf.argmax(output_layer, axis=-1), feed_dict = {input_layer: images})
        acc += np.sum(lbl == lbls)

    print(float(acc) / (test_iters * batch_size))

    # Plot the loss per training batch
    plt.plot(train_losses)
    plt.show()
