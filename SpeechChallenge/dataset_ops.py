"""Calculate the mean and standard deviation of the entire dataset for normalization.
"""

import data_helpers as dataset
import cnn_helpers as cnn
import tensorflow as tf
import numpy as np

def calc_mean_var():
    """Gets a list of all filenames in the training partition
    """
    train_set_file = 'training.txt'
    num_samples = dataset.count_lines(train_set_file)

    files = tf.placeholder(tf.string, [num_samples])
    all_audio = tf.squeeze(tf.map_fn(cnn.load_wav, files, tf.float32))
    mean, var = tf.nn.moments(all_audio, axes=0)
    
    filenames, _, _, _ = dataset.get_filenames(num_samples)
    with tf.Session() as sess:
        m, v = sess.run((mean, var), {files: filenames})
        return m, v

def save_mean_var(filename):
    mean, var = calc_mean_var()
    np.savez(filename, mean=mean, var=var)
