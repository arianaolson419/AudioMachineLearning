"""Implement the CNN used for keyword spotting from the TensorFlow Simple Audio
Recognition Tutorial example. The paper can be found at
http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
"""

# Signal processing and ML libraries
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import tensorflow.contrib.signal as contrib_signal
from tensorflow.python.ops import io_ops
import random

def load_wav(filepath, desired_channels=1, desired_samples=16000):
    """Defines the piece of the computational graph that loads an audio file.

    Parameters
    ----------
    filepath : a tensor of dtype tf.string and shape (1,) representing
        the wav file being loaded.
    desired_channels : the number of channels the wav file should be decoded into.
        Defaults to 1.
    desired_samples : the number of samples to be loaded from the wav file.
        Defaults to 16000.

    Returns
    -------
    a tensor of dtype tf.float32 and shape (1, desired_samples) representing a loaded wav file
    """
    wav_loader = io_ops.read_file(filepath)
    wav_decoder = contrib_audio.decode_wav(
            wav_loader,
            desired_channels=desired_channels,
            desired_samples=desired_samples)
    audio = wav_decoder.audio
    return audio

def load_background_noise(noise_filepath, desired_channels=1):
    """Loads all samples in a given background noise file.
    """
    wav_loader = io_ops.read_file(noise_filepath)
    wav_decoder = contrib_audio.decode_wav(
            wav_loader,
            desired_channels=desired_channels)
    background_noise = wav_decoder.audio
    return background_noise

def get_noise_slice(background_noise, slice_size):
    max_index = tf.shape(background_noise)[0] - slice_size
    start_index = tf.random_uniform([1], 0, max_index, tf.int32)[0]

    noise_slice = background_noise[start_index: start_index + slice_size, :]

    return noise_slice
    
def add_background_noise(audio, noise_slice, background_volume_range):
    background_volume = tf.random_uniform([], 0, background_volume_range, tf.float32)
    noise_volume_adjusted = tf.multiply(background_volume, noise_slice)
    return tf.add(audio, noise_slice)

def compute_logmel_spectrograms(audio, sample_rate, frame_length_seconds, frame_step_seconds):
    """Computes the log-mel spectrograms of a batch of audio clips

    Parameters
    ----------
    audio : a two dimensional tensor of audio samples of shape (num_samples, num_signals)
    sample_rate : the sample rate of the audio signals in Hz
    frame_length_seconds : the width of the STFT, in seconds
    frame_step_seconds : the number of seconds the STFTs are shifted from each other

    Returns
    -------
    A tensor of spectrograms of shape (num_signals, time_units, mel_bins) and dtype tf.float32
    """
    # Convert time parameters to samples
    frame_length_samples = int(frame_length_seconds * sample_rate)
    frame_step_samples = int(frame_step_seconds * sample_rate)

    # Create a spectrogram by taking the magnitude of the Short Time fourier Transform
    stft = contrib_signal.stft(audio, frame_length=frame_length_samples,
            frame_step=frame_step_samples, fft_length=frame_length_samples)
    
    magnitude_spectrograms = tf.abs(stft)

    # Warp the linear scale, magnitude spectrograms into the mel-scale.
    num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 40
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
            upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
            magnitude_spectrograms, linear_to_mel_weight_matrix, 1)

    mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # Compress the mel spectrogram magnitudes.
    log_offset = 1e-6
    log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)

    return log_mel_spectrograms
   # with tf.Session() as sess:
   #     spectrogram_batch = sess.run(log_mel_spectrograms, feed_dict={signals: audio})
   # return spectrogram_batch       

def load_and_process_batch(filepaths, desired_channels=1, desired_samples=16000, frame_length=0.025, frame_width=0.010):
    """Creates a batch of log-mel spectrograms from a list of paths to .wav files.

    Parameters
    ----------
    filepaths : a list of paths to wav files in the dataset.
    desired_channels : the number of channels of audio data to load from the
        .wav files.
    desired_samples : the number of samples to load from each .wav file.
    frame_length : the length in seconds of the STFT frame.
    frame_width : the step size in seconds between STFT frames.

    Returns
    -------
    A tensor of dtype tf.float32 and shape (batch_size, time_bins, mel_bins)
    """
    audio_signals = tf.squeeze(tf.map_fn(load_wav, filepaths, tf.float32))
    spectrograms = compute_logmel_spectrograms(
            audio_signals,
            sample_rate=desired_samples,
            frame_length_seconds=frame_length,
            frame_step_seconds=frame_width)
    return spectrograms
