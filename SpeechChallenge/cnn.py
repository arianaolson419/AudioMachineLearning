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
import dataset_ops as ops

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
    if not os.path.exists('train/audio/_silence_/silence.wav'):
        silence_filename = tf.constant('train/audio/_silence_/silence.wav')
        silence_sample_rate = tf.constant(16000)
        silence_data = tf.zeros((silence_sample_rate, 1), dtype=tf.float32)
        wav_saver = cnn.save_wav(silence_filename, silence_data, silence_sample_rate)

        with tf.Session() as sess:
            sess.run(wav_saver)

    dataset.partition_dataset(FLAGS.validation_percentage, FLAGS.partition_dataset)

    # Recalculate the training set mean and variance if the data is re-partitioned.
    if FLAGS.partition_dataset:
        ops.save_mean_var(FLAGS.mean_var_file)

    # Shuffle datasets while debugging network to get fresh examples (Not using all data for debugging)
    dataset.shuffle_partitioned_files('training.txt')
    dataset.shuffle_partitioned_files('validation.txt')

    # Training parameters
    batch_size = FLAGS.batch_size
    learning_rate = FLAGS.learning_rate

    # Convolution parameters
    num_conv_feature_maps = 54  # n
    conv_weights_height1 = 21# m
    conv_weights_width1 = 8
    conv_weights_height2 = 6
    conv_weights_width2 = 4
    pooling_height = 1  # p
    pooling_width = 3   # q
    stride_height = 1  # s
    stride_width = 1    # v

    he = tf.contrib.keras.initializers.he_normal

    filenames = tf.placeholder(tf.string, [batch_size])
    mean = tf.placeholder(tf.float32, [None])
    var = tf.placeholder(tf.float32, [None])
    labels = tf.placeholder(tf.int32, [batch_size])

    # Define the cnn-one-fpool13 network
    input_layer = tf.expand_dims(cnn.load_and_process_batch(filenames, mean, var), 3)
    conv1 = tf.layers.conv2d(
            inputs=input_layer, 
            filters=num_conv_feature_maps,
            kernel_size=[conv_weights_height1, conv_weights_width1],
            strides=[stride_height, stride_width],
            padding="same")

    layer_norm_1 = tf.contrib.layers.layer_norm(conv1)

    conv2 = tf.layers.conv2d(
            inputs=layer_norm_1,
            filters=num_conv_feature_maps,
            kernel_size=[conv_weights_height2, conv_weights_width2],
            strides=[1, 1],
            padding="same")

    layer_norm_2 = tf.contrib.layers.layer_norm(conv2)

    flat_layer = tf.reshape(layer_norm_2, [-1, 98 * 40 * num_conv_feature_maps])

    output_layer = tf.layers.dense(
            inputs=flat_layer,
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
        
        mean_var = np.load(FLAGS.mean_var_file)
        m = mean_var['mean']
        v = mean_var['var']

        log_dir = FLAGS.log_dir

        for i in range(training_iterations):
            train_pos = 0
            train_pos_unknown = 0
            test_pos = 0
            test_pos_unknown = 0
            print('training iteration: {}'.format(i))
            dataset.shuffle_partitioned_files('training.txt')
            dataset.shuffle_partitioned_files('training_unknown.txt')
            for j in tqdm(range(training_steps)):
                files, batch_labels, new_position, new_position_unknown = dataset.get_filenames(
                        batch_size,
                        train_pos,
                        train_pos_unknown,
                        'training')
                _, lossval = sess.run((opt, loss), feed_dict={filenames: files, mean: m, var: v, labels: batch_labels})
                train_pos = new_position
                train_pos_unknown = new_position_unknown
                if j % 10 == 0:
                    with open(log_dir + '/' + FLAGS.log_file, 'a') as f:
                        f.write('lossval: {}, training_step: {}\n'.format(lossval, i * training_steps + j))


                if j % FLAGS.validation_frequency == FLAGS.validation_frequency - 1:
                    acc = 0
                    dataset.shuffle_partitioned_files('validation.txt')
                    dataset.shuffle_partitioned_files('validation_unknown.txt')
                    for _ in tqdm(range(validation_steps)):
                        files, batch_labels, new_position, new_position_unknown = dataset.get_filenames(
                                batch_size,
                                test_pos,
                                test_pos_unknown,
                                'validation')
                        test_pos = new_position
                        test_pos_unknown = new_position_unknown
                        lbl = sess.run(tf.argmax(output_layer, axis=1), feed_dict={filenames: files, mean: m, var: v})
                        acc += np.sum(lbl == batch_labels)
                        acc_percent = 100 * acc / (batch_size * validation_steps)
                        with open(log_dir + '/' + FLAGS.log_file, 'a') as f:
                            f.write('accuracy: {}, training_step: {}\n'.format(acc_percent, i * training_steps + j))

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
    parser.add_argument('--training_steps', type=int, default=400,
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
    parser.add_argument('--mean_var_file', type=str, default='mean_var.npz',
            help='File containing numpy arrays of the mean and variance of the training set')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
