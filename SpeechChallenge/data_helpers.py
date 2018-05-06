"""Helper functions for initial partitioning and batching of data in the keyword spotting CNN. To be used with the Speech Commands dataset.
"""

import random

# Hashing libraries
import re
import os.path
import hashlib
import sys

# Define categories
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

def partition_dataset(validation_percentage, overwrite=False):
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
                partition = choose_set(filename, validation_percentage)
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
    label_strings = [path.split('/')[2] for path in filepaths]
    get_labels = lambda x: x.split('/')[2]
    legal_labels = lambda x: 'unknown' if get_labels(x) not in legal_commands else get_labels(x)
    labels_to_ints = lambda x: label_dict[legal_labels(x)]
    
    label_ints = list(map(labels_to_ints, filepaths))
    return filepaths, label_ints, new_position
