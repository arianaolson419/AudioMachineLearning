import network
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

input_layer_size = np.shape(iris.data)[1]
output_layer_size = 3

net = network.Network(
        [input_layer_size, 15, 7, output_layer_size], 
        network.sigmoid, 
        network.sigmoid_prime)

iris_results = [[0] * 3 for target in iris.target]

for i, classification in enumerate(iris.target):
    iris_results[i][classification] = 1
print(net.train(iris.data, iris_results, 0.5, 0.1, True, 10000))
print(net.test(iris.data, iris_results, 0.25))

