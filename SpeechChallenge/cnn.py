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
        silence_data = tf.zeros((sample_rate, 1))
        wav_saver = cnn.save_wav(silence_filepath, silence_sample_rate, silence_data)

        with tf.Session() as sess:
            sess.run(wav_saver)

    # Training parameters
    batch_size = FLAGS.batch_size
    learning_rate = FLAGS.learning_rate

    he = tf.contrib.keras.initializers.he_normal

    training_mode = tf.placeholder(tf.bool, ())
    train_categories, validation_categories = dataset.make_labelled_data()
    categories = tf.cond(training_mode, lambda: train_categories, lambda: validation_categories)
    labels = tf.random_uniform([batch_size], 0, len(categories), dtype=tf.int32)
    filenames = dataset.make_labelled_batch(categories, labels)
    mean_var = np.load(FLAGS.mean_var_file)
    mean = tf.constant(mean_var['mean'])
    var = tf.constant(mean_var['var'])


    # Define the cnn-one-fpool13 network
    input_layer = tf.expand_dims(cnn.load_and_process_batch(filenames, mean, var), 3)
    conv1 = tf.layers.conv2d(
            inputs=input_layer, 
            filters=64,
            kernel_size=[8, 8],
            strides=[5, 3],
            activation=tf.nn.leaky_relu,
            padding="same")

    layer_norm_1 = tf.contrib.layers.layer_norm(conv1)

    conv2 = tf.layers.conv2d(
            inputs=layer_norm_1,
            filters=128,
            kernel_size=[8, 6],
            strides=[3, 2],
            activation=tf.nn.leaky_relu,
            padding="same")

    layer_norm_2 = tf.contrib.layers.layer_norm(conv2)

    conv3 = tf.layers.conv2d(
            inputs=layer_norm_2,
            filters=256,
            kernel_size=[5, 5],
            strides=[2, 2],
            activation=tf.nn.leaky_relu,
            padding="same")

    layer_norm_3 = tf.contrib.layers.layer_norm(conv3)
    
    conv4 = tf.layers.conv2d(
            inputs=layer_norm_3,
            filters=512,
            kernel_size=[2, 2],
            strides=[2, 2],
            activation=tf.nn.leaky_relu,
            padding="same")

    layer_4_norm = tf.contrib.layers.layer_norm(conv4)

    conv5 = tf.layers.conv2d(
            inputs=layer_4_norm,
            filters=12,
            kernel_size=[2, 2],
            strides=[2, 2],
            padding="same")

    output_layer = tf.layers.flatten(conv5)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output_layer))
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    prediction = tf.argmax(output_layer, axis=1)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        training_steps = FLAGS.training_steps
        training_iterations = FLAGS.training_iterations
        validation_steps = FLAGS.validation_steps
        log_dir = FLAGS.log_dir

        for i in range(training_iterations):
            print('training iteration: {}'.format(i))
            for j in tqdm(range(training_steps)):
                _, pred, lbl, lossval = sess.run((opt, prediction, labels, loss), {training_mode: True})
                t_acc = np.sum(pred == lbl)
                acc_percent = 100 * t_acc  / (batch_size)

                with open(log_dir + '/' + FLAGS.log_file, 'a') as f:
                    f.write('train_loss: {}, training_step: {}\n'.format(lossval, i * training_steps + j))
                    f.write('train_accuracy: {}, training_step: {}\n'.format(acc_percent, i * training_steps + j))


                if j % FLAGS.validation_frequency == FLAGS.validation_frequency - 1:
                    print('accuracy: {}'.format(t_acc))
                    v_acc = 0
                    for _ in tqdm(range(validation_steps)):
                        pred, lbl, lossval = sess.run((prediction, labels, loss), {training_mode: False})
                        v_acc += np.sum(pred == lbl)
                        acc_percent = 100 * v_acc / (batch_size * validation_steps)
                        with open(log_dir + '/' + FLAGS.log_file, 'a') as f:
                            f.write('validation_loss: {}, training_step: {}\n'.format(lossval, i * training_steps + j))
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
