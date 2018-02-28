import numpy as np
import unittest

class Network(object):
    """A simple fully connected neural network. The network currently uses gradient
    descent as opposed to stochastic gradient descent. Future features will include SGD.
    """
    def __init__(self, neurons_per_layer, activation_function, derivative_function):
        """Initialize a fully connected neural network with layers of a specified
        number of neurons and random Gaussian distributed weights and biases.

        Parameters
        ----------
        neurons_per_layer : a list containing the number of neurons that each
            layer should contain. The first element of the list corresponds to
            the size of the input layer, and so on.
        activation_function : a lambda function that will calculate the activation
            of the neurons. This is the function used in forward propagation.
            The activation function must take a single number value and return a 
            single number value.
        derivative_function : a lambda function that is the derivative of the
            activation function. This is the function used to calculate the 
            gradient during backpropagation. The derivative function must take a single number value as 
            its argument and return a single number value.

        Instance variables
        ------------------
        num_layers : The number of layers in the network.
        activation_function : A lambda function defining the activation function of the neurons.
        derivative_function : A lambda function defining the derivative of the activation function.
        weights : A list of numpy arrays representing the weights of the inputs to the neurons.
            The shape of each array is (k, j), where k is the number of neurons in the current
            layer, and j is the number of neurons in the previous layer. For a weight array w in 
            the weights list, w[k][j] gives the weight of the input from the jth neuron of the
            previous layer to the kth neuron of the current layer.

            The weights are initialized as floating point samples from the standard normal
            distribution.

            There are no weights associated with the input layer, so the list begins with the
            weights of the inputs of the second layer in the network. For example, in a network with
            three layers, of sizes 3, 4, and 2 respectively, the first layer is the input layer.
            net.weights[0] would give a 4x3 numpy array associated with the weights of the inputs to 
            the second layer.
        biases : A list of numpy arrays representing the biases of the iniputs to the neurons.
            The shape of each array is (j, 1), where j is the number of neurons in the current layer.
            For a bias array b in the bias list, b[j] gives the bias of the jth neuron in the
            current layer.

            The biases are initialized as floating point samples from the standard normal 
            distribution.

            There are no biases associated with the input layer, so the list begins with the biases
            of the inputs of the second layer in the network. For example, in a network with three
            layers, of sizes 3, 4, and 2 respectively, the first layer is the input layer.
            net.biases[0] would give a 4x1 numpy array associated with the biases of the inputs to
            the second layer.
        activations : a list containing num_layers numpy arrays containing the activations of the 
            neurons in each layer of the network. The shape of each array is (j, 1), where j is
            the number of neurons in the given layer. 
            
            The first array in the list is associated with the input layer. For an activation array, a, 
            in a given layer, a[j] is the activation of the jth neuron in that layer.

            The activations are all initialized to 0, and are calculated during forward propagation.
        layer_errors : A list containing numpy arrays associated with the error in each layer. The 
            shape of each array is (j, 1), where j is the number of neurons in the given layer.

            There is no error associated with the input layer, so for layer l, the array containing
            the errors in this layer is layer_errors[l - 1]. For a layer error array, e, e[j] is the
            error of the jth neuron in the given layer.            
        """
        self.num_layers = len(neurons_per_layer)
        self.activation_function = activation_function
        self.derivative_function = derivative_function
        self.weights = [
                np.random.randn(
                    neurons_per_layer[i], neurons_per_layer[i-1]) 
                for i in range(1, self.num_layers)]
        self.biases = [np.random.randn(neurons_per_layer[i], 1) 
                for i in range(1, self.num_layers)]
        self.activations = [np.zeros((neurons, 1)) for neurons in neurons_per_layer]
        self.layer_errors = [np.zeros((neurons, 1)) for neurons in neurons_per_layer[1:]]

    def input_data(self, inputs):
        """Assigns the values of the activations of the first layer to the input values.

        Parameters
        ----------
        inputs : A list containing the values of the input layer.
        """
        for i, x in enumerate(inputs):
            self.activations[0][i] = x

    def feedforward(self):
        """Forward propagates the input values through the network."""
        for i in range(1, self.num_layers):
            self.activations[i] = self.compute_activation(i)

    def compute_z(self, layer_index):
        """Computes the inputs to the activation function, z for a given layer.
        
        z = w(l)'a(l-1) + b(l)
        w : The weight matrix associated with the current layer.
        a : The activations associated with the previous layer.
        b : The biases associated with the current layer.

        Parameters
        ----------
        layer_index : the index of the layer for which to compute the activation
        function inputs. The layer index must not be 0, because the input layer
        does not have associated weights and biases.

        Returns
        -------
        The inputs to the activation function for the given layer.
        """
        prev_activation = self.activations[layer_index - 1]
        weights = self.weights[layer_index - 1]
        biases = self.biases[layer_index - 1]
        print("wa", layer_index, np.shape(weights), np.shape(prev_activation))
        return np.matmul(weights, prev_activation) + biases

    def compute_activation(self, layer_index):
        """ Computes the activations for a given layer.

        Parameters
        ----------
        layer_index : The index of the layer for which to compute the activation
        function. The layer index must not be 0, because the input layer
        does not have associated weights and biases.

        Returns
        -------
        The activations of the given layer.
        """
        z = self.compute_z(layer_index)
        return self.activation_function(z)

    def compute_cost(self, expected_output):
        """Compute the quadratic cost function for the outputs

        C = 0.5||y - a||^2
        y : The expected output.
        a : The activations of the output layer        

        Parameters
        ----------
        expected_output : The expected output for the inputs to the network.
        
        Returns
        -------
        A value representing the cost for the given training example.
        """
        actual_output = self.activations[-1]
        diff = np.reshape(expected_output, np.shape(actual_output)) - actual_output
        norm = np.linalg.norm(diff)
        return 0.5 * np.square(norm)

    def compute_cost_gradient(self, expected_output):
        """Compute the cost gradient for a given training example.

        Parameters
        ----------
        expected_output : The expected output for the inputs to the network.

        Returns
        -------
        A numpy array representing the cost gradient for the given training example.
        """
        actual_output = self.activations[-1]
        return np.reshape(expected_output, np.shape(actual_output)) - actual_output

    def compute_output_error(self, expected_output):
        """Compute the error in the output layer.

        error = cost_gradient * derivative_function(z)

        Returns
        -------
        A numpy array representing the error in the output layer.
        """
        z_output = self.compute_z(self.num_layers - 1)
        cost_gradient = self.compute_cost_gradient(expected_output)
        return cost_gradient * self.derivative_function(z_output)

    def compute_layer_error(self, future_error, future_weights, layer_zs):
        """Compute the error in the hidden layers of the network.

        Parameters
        ----------
        future_error : The error of the next layer in the network.
        future_weights : The weights of the next layer
        layer_zs : The inputs to the activation function for the current layer.

        Returns
        -------
        A numpy array representing the error in the given hidden layer.
        """
        future_error = self.layer_errors[layer_index - 1]
        future_weights = self.weights[layer_index - 1]
        layer_zs = compute_z(layer_index)
        return np.matmul(np.transpose(future_weights), future_error) * self.derivative_function(layer_zs)
 
    def backprop(self, expected_output):
        """Performs backpropagation through the network
        
        Parameters
        ----------
        expected_output : a numpy array representing the expected output
        of the neural network for a given training example.
        """
        self.layer_errors[-1] = self.compute_output_error(expected_output)
        for i in range(len(self.layer_errors) - 1, 0, -1):
            layer_zs = self.compute_z(i)
            print(i)
            self.layer_errors[i] = self.compute_layer_error(self.layer_errors[i], self.weights[i], layer_zs)

    def calculate_gradients(self, layer_index):
        """Calculates the gradient with respect to each of the weights and biases of the network for a given layer.

        Parameters
        ----------
        layer_index : The index of the layer for which to calculate the gradients. The layer index
        may not be 0, because the gradient cannot be calculated for the input layer.

        Returns
        -------
        a tuple containing two numpy arrays. The first element of the tuple contains the gradient with
        respect to the biases of the layer. The second element containes the gradient with respect to the
        weights of the layer.
        """
        prev_activations = self.activations[layer_index-1]
        layer_error = self.layer_errors[layer_index-1]
        bias_gradient = layer_error
        weight_gradient = prev_activations * layer_error
        return (bias_gradient, weight_gradient)

    def update_weights_and_biases(self, learning_rate):
        """Performs gradient descent to update the weights and biases of the network.

        Parameters
        ----------
        learning_rate : The rate at which to adjust the weights and biases in the direction of
            the gradient.
        """

        print([np.shape(weights) for weights in self.activations], [np.shape(errors) for errors in self.layer_errors])
        for i in range(0, len(self.weights)):
            bias_gradient, weight_gradient = self.calculate_gradients(i + 1)
            self.biases[i] -= learning_rate * bias_gradient
            self.weights[i] -= learning_rate * weight_gradient

    def train(self, training_data, expected_result, learning_rate, tolerance):
        """Trains the network using gradient descent for a given set of training data
        and expected results.

        Parameters
        ----------
        training_data : A list of inputs to the network.
        expected_result : A list of expected outputs for each training example.
        learning_rate : A value representing the step size to take during gradient descent.
        tolerance : The cost at which the network can stop training.

        Returns
        -------
        average_cost_array : A numpy array containing the average cost of each run through the
            examples.
        """
        average_cost = tolerance + 1
        average_cost_array
        while average_cost > tolerance:
            sum_cost = 0
            for example, output in zip(training_data, expected_result):
                self.input_data(example)
                self.feedforward()
                sum_cost += self.compute_cost(output)
                self.backprop(output)
                self.update_weights_and_biases(learning_rate)
            average_cost = sum_cost / len(expected_result)
            average_cost_array.append(average_cost)
        return average_cost_array

    def test(self, testing_data, expected_result, threshold):
        """Tests the network on a set of testing data with the network's
        current weights and biases.

        Parameters
        ----------
        testing_data : A list of inputs to the network for each testing example.
        expected_result : A list of exppected outputs from the network for each testing example.
        threshold : The maximum cost at which a classification can be considered correct.

        Returns
        -------
        (accuracy, classifications)
        accuracy : The accuracy of the classifications. This is a fraction of the number
            of examples classified correctly out of the total number of testing examples.
        classifications : a list containing tuples of the outputs for each example and
            an indicator of whether or not the example was correctly classified. The indicator
            will be 1 if the example wasa correctly classified and a 0 otherwise.
            example tuple: ([0.01, 0.9, 0.05], 1)
        """
        num_correct = 0
        classifications = []
        for example, output in zip(testing_data, expected_output):
            self.input_data(example)
            self.feedforward()
            if self.compute_cost(output) <= threshold:
                correct_classification = 1
                num_correct += 1
            else:
                correct_classification = 0
            classifications.append(self.activations[-1], correct_classification)
        accuracy = num_correct / len(testing_data)
        return (accuracy, classifications)

# Define the sigmoid function that may be input to the network
# as its activation function.
sigmoid = lambda z: 1 / (1 + np.exp(-z))

# Define the derivative of the sigmoid function that may be
# input to the network as its derivative function.
sigmoid_prime = lambda z: sigmoid(z) * (1 - sigmoid(z))
