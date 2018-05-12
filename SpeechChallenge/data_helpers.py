"""Helper functions for initial partitioning and batching of data in the keyword spotting CNN. To be used with the Speech Commands dataset.
"""

import random
import math

import tensorflow as tf

# Hashing libraries
import re
import os.path
import pathlib
import hashlib
import sys

# Define categories
all_commands = 'yes no up down left right on off stop go zero one two three' + \
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
    speaker_id, _, _ = re.split('[_]', filename)
    speaker_id = speaker_id.encode('utf-8')
    hashed_speaker_id = hashlib.sha1(speaker_id).hexdigest()
    percentage_hash = ((int(hashed_speaker_id, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        return 'validation'
    else:
        return 'training'

def make_labelled_batch(categories, index_sequence):
    """
    Given 'categories', a list of tensors of possibly non-uniform size, and 'index_sequence', 
    a sequence of indices into the 'categories' list, returns a tensor whose ith element is a 
    value randomly selected from the tensor at 'categories[index_sequence[i]]'.
    """
    assert len(index_sequence.shape) == 1
    tensors = tf.TensorArray(
        dtype = tf.string,
        size = len(categories),
        dynamic_size = False,
        clear_after_read = False,
        infer_shape = False,
    )
    for i, filename_tensor in enumerate(categories):
        tensors = tensors.write(i, filename_tensor)
    def get_file_name_at(idx):
        category = tensors.read(idx)
        return category[tf.random_uniform([], 0, tf.shape(category)[0], dtype=tf.int32)]
    mapped = tf.map_fn(get_file_name_at, index_sequence, dtype=tf.string)
    if index_sequence.shape[0] is not None:
        mapped.set_shape(index_sequence.shape)
    return mapped

def make_labelled_data():
    import tensorflow as tf
    train_data = [[] for _ in label_dict]
    validation_data = [[] for _ in label_dict]

    legal = frozenset(legal_commands)
    rootdir = pathlib.Path('train') / 'audio'
    for d in rootdir.iterdir():
        if d.name == '_background_noise_':
            continue
        label = d.name if d.name != '_silence_' else 'silence'
        try:
            idx = label_dict[label]
        except KeyError:
            idx = label_dict['unknown']
        for wavfile in d.iterdir():
            if wavfile.name[-4:] != '.wav':
                continue
            if wavfile.name == 'silence.wav':
                train_data[label_dict['silence']].append(wavfile)
                validation_data[label_dict['silence']].append(wavfile)
                continue
            if choose_set(wavfile.name) == 'training':
                train_data[idx].append(wavfile)
            elif choose_set(wavfile.name) == 'validation':
                validation_data[idx].append(wavfile)
    train_tensors = [tf.constant([str(path) for path in cat], dtype=tf.string) for cat in train_data]
    validation_tensors = [tf.constant([str(path) for path in cat], dtype=tf.string) for cat in validation_data]
    return train_tensors, validation_tensors

