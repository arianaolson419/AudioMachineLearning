"""Plots the output log information from the CNN.
"""

import matplotlib.pyplot as plt
import re
import argparse

FLAGS = None

def read_file(log_file):
    """Read the lines of the file and processes the contents into numerical values.

    Parameters
    ----------
    log_file : the file containing output log information from the CNN. The
        contents of the file are assumed to have the following format:
        - The file contains loss values and accuracies of the network as well
          as the training step at which these values were recorded.
        - Accuracies: accuracy: <float> training step: <int>
        - Losses: lossval: <float> training step: <int>

    Returns
    -------
    A dictionary with keys "lossval" and "accuracy", Each with values as lists.
    lossval : A list of tuples. The first element of the tuple is the training
        step of the logged loss value, and the second is the loss value logged.
    accuracy : A list of tuples. The first element of the tuple is the training
        step of the logged accuracy, and the second is the loss value logged.
    """
    with open(log_file, 'r') as f:
        lines = f.readlines()

    sub_chars = ['\n', ':', ',']
    lines = map(lambda x: re.sub('|'.join(sub_chars), '', x), lines)

    lossval = []
    accuracy = []
    data = {'lossval': [], 'accuracy': []}

    for line in lines:
        elements = line.split(' ')
        data[elements[0]].append((elements[3], elements[1]))
    return data

def plot_data(data):
    loss_x = [loss[0] for loss in data['lossval']]
    loss_y = [loss[1] for loss in data['lossval']]

    acc_x = [acc[0] for acc in data['accuracy']]
    acc_y = [acc[1] for acc in data['accuracy']]

    plt.plot(loss_x, loss_y)
    plt.plot(acc_x, acc_y)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file_path', type=str, default='cnn_output_logs/cnn_output.txt',
            help='The path to the output file to plot')
    FLAGS, unparsed = parser.parse_known_args()
    data = read_file(FLAGS.log_file_path)
    plot_data(data)
