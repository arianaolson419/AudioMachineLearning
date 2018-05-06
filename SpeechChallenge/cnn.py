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
from tqdm import tqdm

# Command line flags
import sys
import os
import argparse

FLAGS = None

def train():
    dataset.partition_dataset(FLAGS.validation_percentage, FLAGS.partition_dataset)
    # Shuffle datasets while debugging network to get fresh examples (Not using all data for debugging)
    dataset.shuffle_partitioned_files('training.txt')
    dataset.shuffle_partitioned_files('validation.txt')

    # Training parameters
    batch_size = FLAGS.batch_size
    learning_rate = FLAGS.learning_rate

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

        training_steps = FLAGS.training_steps
        training_iterations = FLAGS.training_iterations
        validation_steps = FLAGS.validation_steps
        
        log_dir = FLAGS.log_dir
#        train_losses = []
#        accuracies = []

        for i in range(training_iterations):
            train_pos = 0
            test_pos = 0
            print('training iteration: {}'.format(i))
            for j in tqdm(range(training_steps)):
                files, batch_labels, new_position = dataset.get_filenames(batch_size, train_pos, 'training')
                _, lossval = sess.run((opt, loss), feed_dict={filenames: files, labels: batch_labels})
                train_pos = new_position
                if j % 10 == 0:
                    with open(log_dir + '/' + FLAGS.log_file, 'a') as f:
                        f.write('lossval: {}, training_step: {}\n'.format(lossval, i + j))


                if j % FLAGS.validation_frequency == 0:
                    acc = 0
                    for _ in tqdm(range(validation_steps)):
                        files, batch_labels, new_position = dataset.get_filenames(batch_size, test_pos, 'validation')
                        lbl = sess.run(tf.argmax(output_layer, axis=1), feed_dict={filenames: files})
                        acc += np.sum(lbl == batch_labels)
                        acc_percent = acc / (batch_size * validation_steps)
                        with open(log_dir + '/' + FLAGS.log_file, 'a') as f:
                            f.write('accuracy: {}, training_step: {}\n'.format(acc_percent, i + j))

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100,
            help='The number of audio clips in a training batch')
    parser.add_argument('--validation_percentage', type=int, default=10,
            help='The percent of the labelled data devoted to validation. The'
            + 'remaining data is partitioned into the training set.')
    parser.add_argument('--training_iterations', type=int, default=10,
            help='The number of iterations through the data during training')
    parser.add_argument('--training_steps', type=int, default=1000,
            help='The number of steps per training iteration. Each step'
            + 'processes one batch of data')
    parser.add_argument('--validation_steps', type=int, default=1,
            help='The number of steps used during validation. One batch is'
            + 'processed during each validation step.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
            help='The initial learning rate of the optimizer')
    parser.add_argument('--log_dir', type=str, default=os.path.join(os.getenv('PWD'), 'cnn_output_logs'),
            help='Network performance log directory')
    parser.add_argument('--log_file', type=str, default='cnn_output.txt',
            help='File to write output logs to')
    parser.add_argument('--partition_dataset', type=bool, default=False,
            help='If true, partitions data in train/audio into "training.txt"'
            + 'and "validation.txt" and overwrites the existing txt files.'
            + 'Otherwise, the network will use the existing partitions')
    parser.add_argument('--validation_frequency', type=int, default=100,
            help='The number of training steps between validation steps.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
