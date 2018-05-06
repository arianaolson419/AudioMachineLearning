"""Defines a convolutional neural network for keyword speech recognition and
trains it on the dataset provided by the Kaggle TensorFlow Speech Recognition
Challenge, which is a modified version of the Speech Commands dataset.

The CNN is based on the architecture defined in "Convolutional Neural Networks
for Small-footprint Keyword Spotting" which can be found at
https://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
"""

# Helper functions
import cnn_helpers as cnn
import data_helpers as dataset

# Machine learning libraries
import tensorflow as tf
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt

from tqdm import tqdm

# Shuffle datasets while debugging network to get fresh examples (Not using all data for debugging)
dataset.shuffle_partitioned_files('training.txt')
dataset.shuffle_partitioned_files('validation.txt')

# Training parameters
batch_size = 100
learning_rate = 0.001

# Convolution parameters
num_conv_feature_maps = 54  # n
conv_weights_height = 98# m
conv_weights_width = 8
pooling_height = 1  # p
pooling_width = 3   # q
stride_height = 1  # s
stride_width = 1    # v

he = tf.contrib.keras.initializers.he_normal

filenames = tf.placeholder(tf.string, [batch_size])
labels = tf.placeholder(tf.int32, [batch_size])

# Define the cnn-one-fpool13 network
input_layer = tf.expand_dims(cnn.load_and_process_batch(filenames), 3)
print(input_layer.shape)
conv1 = tf.layers.conv2d(
        inputs=input_layer, 
        filters=num_conv_feature_maps,
        kernel_size=[pooling_height, pooling_width],
        strides=[stride_height, stride_width],
        padding="same")

max_pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[pooling_height, pooling_width],
        strides=[stride_height, stride_width],
        padding="same")

pool1_flat = tf.reshape(max_pool1, [-1, 98 * 40 * num_conv_feature_maps])

linear_layer = tf.layers.dense(
        inputs=pool1_flat,
        units=32,
        activation=tf.nn.relu,
        kernel_initializer=he(),
        bias_initializer=he())

dense_layer1 = tf.layers.dense(
        inputs=linear_layer,
        units=128,
        activation=tf.nn.relu,
        kernel_initializer=he(),
        bias_initializer=he())

dense_layer2 = tf.layers.dense(
        inputs=dense_layer1,
        units=128,
        activation=tf.nn.relu,
        kernel_initializer=he(),
        bias_initializer=he())

output_layer = tf.layers.dense(
        inputs=dense_layer2,
        units=len(dataset.all_categories),
        activation=tf.nn.softmax,
        kernel_initializer=he(),
        bias_initializer=he())

labels = tf.placeholder(shape=(batch_size), dtype=tf.int32)
loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=output_layer, labels=labels))
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Size of datasets (not including added silence in batches)
    training_examples = dataset.count_lines('training.txt')
    validation_examples = dataset.count_lines('validation.txt')

    num_batches = 1
    training_iterations = 1
    validation_size = 5
    
    output_file = 'cnn_progress.out'
    train_losses = []
    acc = 0

    for i in range(training_iterations):
        train_pos = 0
        test_pos = 0
        print('training iteration: {}'.format(i))
        for j in tqdm(range(num_batches)):
            files, batch_labels, new_position = dataset.get_filenames(batch_size, train_pos, 'training')
            _, lossval = sess.run((opt, loss), feed_dict={filenames: files, labels: batch_labels})
            train_pos = new_position

            if j % 10 == 0:
                with open(output_file, 'a') as f:
                    f.write('lossval: {}, training iteration: {}'.format(lossval, i))

        for _ in tqdm(range(validation_size)):
            files, batch_labels, new_position = dataset.get_filenames(batch_size, test_pos, 'validation')
            lbl = sess.run(tf.argmax(output_layer, axis=1), feed_dict={filenames: files})
            acc += np.sum(lbl == batch_labels)
        print('accuracy: {}'.format(acc / (batch_size * validation_size)))
    plt.plot(train_losses)
    plt.show()
