# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def display_image(image_vector):
    """Reshapes an image vector into a 28x28 numpy array
    and displays the result as a matplotlib figure.

    Parameters
    ----------
    image_vector : A vector representing the pixels of a 28x28
        image. The pixel values range from 0 to 255, where 0 is
        black and 255 is white. The pixels in the vector must be
        arranged such that a pixel at index k is equal to 28 * i + j
        where i is the row and j is the column of the image where the
        pixel is located. i and j are both numbers between 0 and 27.
        see https://www.kaggle.com/c/digit-recognizer/data for more
        information.
    """
    image_matrix = np.reshape(image_vector, (28, 28))

    # Plot the image matrix
    plt.imshow(image_matrix, cmap='gray')
    plt.show()

if __name__ == "__main__":
    # Read the data from the csv files.
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')

    X_train = train_data.drop('label', axis=1)
    Y_train = train_data['label']
    X_test = test_data
    
    knn = KNeighborsClassifier(n_neighbors = 15)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    acc_knn = knn.score(X_train, Y_train)
    print(acc_knn)
