"""Implement the CNN used for keyword spotting from the TensorFlow Simple Audio
Recognition Tutorial example. The paper can be found at
http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
"""

# Signal processing and ML libraries
import numpy as np
import tensorflow as tf
import tensorflow.contrib.signal as contrib_signal
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
import random


# Visualization libraries
import matplotlib.pyplot as plt

# Hashing libraries
import re
import os.path
import sys
import hashlib

# Profiling
import cProfile

# Define categories
# TODO: these lists are also defined in cnn.py. Consolidate into one file.
all_commands = 'yes no up down left right on off srip go zero one two three' + \
    'four five six seven eight nine bed bird cat dog happy house marvin' + \
    'sheila tree wow'
all_commands = all_commands.split()
legal_commands = 'yes no up down left right on off stop go'.split()
other_categories = 'unknown silence'.split()
all_categories = list(legal_commands) + list(other_categories)
label_dict = {}
for i, cat in enumerate(all_categories):
    label_dict[cat] = i

# Partition data into training and validation sets
MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1    # ~134 million
def choose_set(filename, validation_percentage=10):
    """Choose whether a given file in the dataset should be partitioned into
    the training or validation set.

    Parameters
    ----------
    filename : the name of the file in the dataset in the form <command>/<audio clip>.wav.
    validation_percentage : the desired percentage of the dataset to be set
        aside for validation.

    Returns
    -------
    a string, either 'training' or 'validation', indicating
        which set the file should be partitioned into.
    """
    filename = filename.strip('.wav')
    _, speaker_id, _, _= re.split('[_/]', filename)
    speaker_id = speaker_id.encode('utf-8')
    hashed_speaker_id = hashlib.sha1(speaker_id).hexdigest()
    percentage_hash = ((int(hashed_speaker_id, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        return 'validation'
    else:
        return 'training'

def partition_data_set(overwrite=False):
    """Writes the names of all of the files in the dataset to either
    'training.txt' or 'validation.txt'.

    Parameters
    ----------
    overwrite : a boolean indicating whether or not to overwrite existing text
        files. Defaults to False.
    """
    base = 'train/audio/'
    if os.path.exists('training.txt'):
        if overwrite:
            os.remove('training.txt')
        else:
            return
    if os.path.exists('validation.txt'):
        if overwrite:
            os.remove('validation.txt')
        else:
            return
    for command in all_commands:
        path = base + command
        for _, _, clips in os.walk(path):
            for clip in clips:
                filename = command + '/' + clip
                partition = choose_set(filename)
                destination_file = partition + '.txt'
                with open(destination_file, 'a') as f:
                    f.write(filename + '\n')
    shuffle_partitioned_files('training.txt')
    shuffle_partitioned_files('validation.txt')

def shuffle_partitioned_files(partition):
    """'training.txt' and 'validation.txt' are written to in order, so they
    must be shuffled before being input into the network.

    Parameters
    ----------
    partition : the partitioned file to shuffle. Either 'training.txt' or
        'validation.txt'.
    """
    with open(partition, 'r') as f:
        data = [(random.random(), line) for line in f]
    data.sort()
    with open(partition, 'w') as f:
        for _, line in data:
            f.write(line)

def count_lines(filename):
    """Counts the number of lines in a txt file, corresponding to the number of
    files partitioned into that dataset.

    Parameters
    ----------
    filename : A file representing the partitioned dataset. Either
        'trainning.txt' or 'validation.txt'.

    Returns
    -------
    line_count : the number of lines in the file.
    """
    with open(filename, 'r') as f:
        line_count = 0
        for line in f:
            line_count += 1
    return line_count

def add_background_noise(audio, sample_rate, mode='all'):
    """Add background noise to one second audio clips. It has been shown that
    adding some additional background noise increases the accuracy of the
    network. There are four recorded noise clips, and two generated noise
    clips.

    Parameters
    ----------
    audio : the audio clip to add noise to
    sample_rate : the sample rate of the audio clip
    mode : the origin of the background noise. Either 'all', 'recorded', or 'generated'

    Returns
    -------
    mixed_audio : a one second clip of the audio combined with the background noise.
    """
    signal_noise_ratio_db = np.random.randint(-5, 11)  #dB
    signal_noise_ratio = np.power(10, signal_noise_ratio_db * 0.1)
    background_sample_rate, background_noise = load_random_background_noise(samples=sample_rate, mode=mode)
    print(audio.dtype, audio.shape)
    print(background_noise.dtype, background_noise.shape)
    mixed_audio = background_noise * signal_noise_ratio + audio
    # TODO: this adds way too much noise. figure out how to make it work correctly.
    return mixed_audio
    
def get_filenames(batch_size, file_pointer=0, mode='training'):
    """Gets a list of paths to files in one of the partitioned dataset text files,
    training.txt or validation.txt.

    Parameters
    ----------
    batch_size : the number of audio clips to return. Will return as many as
        possible if there are fewer than batch_size clips left in the file.
    file_pointer : the location in the file in which to start reading lines.
    mode : a string, either 'training' or 'validation'. This determines which
        partitioned dataset the audio clips will be read from. Defaults to
        'training'.

    Returns
    -------
    filepaths, labels, new_position
    filepaths : a list of paths to files from the desired training set
    new_position : the number of bytes into the partitoned dataset file where the functon stopped reading.
    """
    with open(mode + '.txt') as f:
        base = 'train/audio/'
        f.seek(file_pointer)
        filepaths = []
        for _ in range(batch_size):
            filepaths.append(base + f.readline().strip('\n'))
        new_position = f.tell()
    label_strings = [path.split('/')[2] for path in filenames]
    get_labels = lambda x: x.split('/')[2]
    legal_labels = lambda x: 'unknown' if get_labels(x) not in legal_commands else get_labels(x)
    labels_to_ints = lambda x: label_dict[legal_labels(x)]
    
    label_ints = list(map(labels_to_ints
    return filepaths, labels, new_position

# TODO: make this part of the audio processing workflow. Don't return real values!
def load_random_background_noise(samples=16000, mode='all'):
    """Loads a clip of random background noise from the set of available noises.

    Parameters
    ----------
    samples : The number of samples of background noise desired. Defaults to
        1600, or one second of audio at 1600Hz.
    mode : One of 'generated', 'recorded', or 'all'. If the mode is
        'generated', the background noise loaded will be from one of the files of
        computer generated noise. If the mode is 'recorded', the background noise
        loaded will be from one of the files of recorded noise. If the mode is
        'all', the background noise can be from any of the available files.
        Defaults to 'all'.

    Returns
    -------
    sample_rate : the sample rate of the audio, in samples per second.
    background_noise : an audio clip of one of the randomly chosen background noise files from the set of options.
    """
    base_path = '_background_noise_/'
    recorded_noise_files = ['doing_the_dishes.wav', 'dude_miaowing.wav', 'exercise_bike.wav', 'running_tap.wav']
    generated_noise_files = ['pink_noise.wav', 'white_noise.wav']
    all_noise_files = list(recorded_noise_files + generated_noise_files)

    if mode == 'all':
        noise_file = random.choice(all_noise_files)
    elif mode == 'recorded':
        noise_file = random.choice(recorded_noise_files)
    elif mode == 'generated':
        noise_file = random.choice(generated_noise_files)

    sample_rate, full_noise, _ = load_audio_clip(base_path + noise_file)
    start_index = np.random.randint(0, full_noise.shape[0] - samples)
    return sample_rate, np.squeeze(full_noise[start_index:start_index + samples])

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
    A tensor of spectrograms of shape (num_signals, time_units, mel_bins)
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

def load_and_process_batch(filepaths, network_mode='training', desired_channels=1, desired_samples=16000, frame_length=0.025, frame_width=0.010):
    """Creates a batch of log-mel spectrograms from a list of paths to .wav files.

    Parameters
    ----------
    filepaths : a list of paths to wav files in the dataset.
    network_mode : Either 'training' or 'validation', representing the
        partition from which to load files.
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
