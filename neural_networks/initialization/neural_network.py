# L layers neural network


"""
    *** Neural Network Structure ***

    layers     : input - [L-1] hidden - output
    activations:         [L-1] Relu   - sigmoid

    cost function: 
        logprobs = np.multiply(-np.log(AL),Y) + np.multiply(-np.log(1 - AL), 1 - Y)
        cost = 1./m * np.nansum(logprobs)
"""


import numpy as np

from forward_propagation import forward_propagation
from compute_cost import compute_cost
from backpropagation import backward_propagation
from update_parameters import update_parameters

from initialization import initialize_parameters


# Neural Network Model
def model(X, Y, layers_dims, learning_rate=0.01, initialization='random', init_const=0.01, num_of_iterations=10000, print_cost=True, print_cost_after=1000, seed=None):
    L = len(layers_dims) - 1 # number of layers

    # Initialize parameters
    parameters = initialize_parameters(layers_dims, initialization, init_const, seed)

    # Gradient Descent
    for i in range(num_of_iterations):
        # Forward propagation
        AL, caches = forward_propagation(X, parameters, L)

        # Compute cost
        cost = compute_cost(AL, Y)

        # Backward propagation
        grads = backward_propagation(AL, Y, caches)

        # Updating parameters
        parameters = update_parameters(parameters, grads, learning_rate, L)

        # Priniting cost after given iterations
        if print_cost and i % print_cost_after == 0:
            print ("Cost after iteration %i: %f" % (i, cost))

    return parameters
