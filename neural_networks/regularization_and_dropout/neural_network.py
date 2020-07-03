# Neural Network model with `regularization` and `dropout`


"""
    Code is designed in such a way that you can only do 
    dropout or regularization (both cannot be applied together).
    
    Also dropout is applied to only those layers which are 
    having `RELU` activation function.
"""


"""
    *** Neural Network Structure ***

    layers     : input - [L-1] hidden - output
    activations:         [L-1] Relu   - sigmoid

    cost function: 
        logprobs = np.multiply(-np.log(AL),Y) + np.multiply(-np.log(1 - AL), 1 - Y)
        cost = 1./m * np.nansum(logprobs)
"""


import numpy as np

from initialization import initialize_parameters
from forward_propagation import forward_propagation
from compute_cost import compute_cost
from backpropagation import backward_propagation
from update_parameters import update_parameters

from regularization import compute_cost_with_regularization, backward_propagation_with_regularization
from dropout import forward_propagation_with_dropout, backward_propagation_with_dropout


# Neural Network Model
def model(X, Y, layers_dims, learning_rate=0.01, initialization='he', _lambda=0, keep_prob=1, init_const=0.01, num_of_iterations=10000, print_cost=True, print_cost_after=1000, seed=None):
    L = len(layers_dims) - 1 # number of layers

    # Initialize parameters
    parameters = initialize_parameters(layers_dims, initialization, init_const, seed)

    # Gradient Descent
    for i in range(num_of_iterations):
        # Forward propagation
        if keep_prob == 1:
            AL, caches = forward_propagation(X, parameters, L)
        elif keep_prob < 1:
            AL, caches = forward_propagation_with_dropout(X, parameters, L, keep_prob)

        # Compute cost
        if _lambda == 0:
            cost = compute_cost(AL, Y)
        else:
            cost = compute_cost_with_regularization(AL, Y, parameters, _lambda, L)

        # Backward propagation
        if _lambda == 0 and keep_prob == 1:
            grads = backward_propagation(AL, Y, caches)
        elif _lambda != 0:
            grads = backward_propagation_with_regularization(AL, Y, caches, _lambda)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(AL, Y, caches, keep_prob)

        # Updating parameters
        parameters = update_parameters(parameters, grads, learning_rate, L)

        # Priniting cost after given iterations
        if print_cost and i % print_cost_after == 0:
            print ("Cost after iteration %i: %f" % (i, cost))

    return parameters