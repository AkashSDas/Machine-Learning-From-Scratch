import numpy as np

from initialization import initialize_parameters
from forward_propagation import forward_propagation
from compute_cost import compute_cost
from backpropagation import backward_propagation

from regularization import compute_cost_with_regularization, backward_propagation_with_regularization
from dropout import forward_propagation_with_dropout, backward_propagation_with_dropout
from gradient_checking import gradient_checking

from update_parameters import update_parameters_using_gd


# ===========================================
# ### Batch Gradient Descent ###
# ===========================================


# Batch Gradient Descent
def model_using_gd(X, Y, layers_dims, learning_rate=0.01, initialization='random', _lambda=0, keep_prob=1, init_const=0.01, num_of_iterations=10000, print_cost=True, print_cost_after=1000, seed=None):
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
        parameters = update_parameters_using_gd(parameters, grads, learning_rate, L)

        # Priniting cost after given iterations
        if print_cost and i % print_cost_after == 0:
            print ("Cost after iteration %i: %f" % (i, cost))

    # Gradient checking
    gradient_checking(parameters, grads, X, Y, layers_dims, _lambda=_lambda)

    return parameters

