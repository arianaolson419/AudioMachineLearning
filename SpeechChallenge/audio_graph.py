import tensorflow as tf
import cnn_helpers as cnn

filepaths = tf.placeholder(tf.string, [])

signal_list = []
for filepath in filepaths:
    audio = cnn.load_audio_tf(filepath)
    signal_list.append(audio)

audio_signals = tf.stack(signal_list)
spectrogram = cnn.compute_logmel_spectrograms(audio_signals, 16000, 0.025, 0.010)

with tf.Session() as sess:
    sess.run(spectrogram, feed_dict={filepath: 'train/audio/stop/01d22d03_nohash_0.wav'})
