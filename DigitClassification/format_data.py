import numpy as np
import pandas as pd

def load_train_and_validation_data(validation_size=10000):
    """Load data from the training set and partition into training and
    validation data

    Parameters
    ----------
    validation_size : the number of data points to be used for
        validation. These will be taken from the end of the dataset.
    
    Returns
    -------
    A tuple of the form 
        (train_data, train_labels, valid_data, valid_labels)
    """

    train = pd.read_csv('data/train.csv')
    labels = train.pop('label')
    
    train_data = np.array(train[:-validation_size])
    valid_data = np.array(train[-validation_size:])

    train_labels = np.array(labels[:-validation_size])
    valid_labels = np.array(labels[-validation_size:])

    return (train_data, train_labels, valid_data, valid_labels)
    
def load_test_data():
    """Load data from the testing set.

    Returns
    -------
    A numpy array containing the testing data. Note that the
    testing dataset does not provide labels.
    """
    test_data = np.array(pd.read_csv('data/test.csv'))
    return test_data

if __name__ == "__main__":
    train_data, train_labels, validate_data, validate_labels = load_train_and_validation_data()
    test_data = load_test_data()

    np.savez_compressed('data/all_data',
            train_data=train_data,
            train_labels=train_labels,
            validate_data=validate_data,
            validate_labels=validate_labels,
            test_data=test_data)
