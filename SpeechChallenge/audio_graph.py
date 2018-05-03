import numpy as np
from tqdm import tqdm
import tensorflow as tf
import cnn_helpers as cnn

batch_size = 100
filepaths = tf.placeholder(tf.string, [batch_size])

spectrogram = cnn.load_and_process_batch(filepaths)
print(spectrogram.shape)

with tf.Session() as sess:
    position = 0
    for _ in range(5):
        for _ in tqdm(range(10)):
            files, new_position = cnn.get_filenames(batch_size, position)
            spectrograms = sess.run(spectrogram, feed_dict={filepaths: files})
            postion = new_position

