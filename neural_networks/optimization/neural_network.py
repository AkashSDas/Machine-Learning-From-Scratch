# Neural Network model using different optimizers


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
from gradient_checking import gradient_checking

from update_parameters import initialize_velocity, initialize_adam
from update_parameters import update_parameters_using_gd, update_parameters_using_momentum, update_parameters_using_adam
from mini_batch_gradient_descent import random_mini_batches


# ===========================================
# ### Neural Network ###
# ===========================================

def model(X, Y, layers_dims, learning_rate=0.01, optimizer='adam', beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, mini_batch_size=64, initialization='random', _lambda=0, keep_prob=1, init_const=0.01, num_of_iterations=10000, print_cost=True, print_cost_after=1000):
    L = len(layers_dims) - 1 # number of layers
    costs = []               # to keep track of total cost
    seed = 10                # For grading purposes, so that your "random" minibatches are the same as ours
    t = 0                    # initializing the counter required for Adam update
    m = X.shape[1]           # number of training example

    # Initialize parameters
    parameters = initialize_parameters(layers_dims, initialization, init_const, seed)

    # Initialize the optimizer
    if optimizer == 'gd':
        pass # no initialization required for gradient descent
    elif optimizer == 'momentum':
        v = initialize_velocity(parameters, L)
    elif optimizer == 'adam':
        v, s = initialize_adam(parameters, L)

    # Optimization loop
    for i in range(num_of_iterations):
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        mini_batches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0

        for mini_batch in mini_batches:
            # Select a minibatch
            (minibatch_X, minibatch_Y) = mini_batch

            # Forward propagation
            if keep_prob == 1:
                AL, caches = forward_propagation(minibatch_X, parameters, L)
            elif keep_prob < 1:
                AL, caches = forward_propagation_with_dropout(minibatch_X, parameters, L, keep_prob)

            # Compute cost and add to the total cost
            if _lambda == 0:
                cost_total += compute_cost(AL, minibatch_Y)
            else:
                cost_total += compute_cost_with_regularization(AL, minibatch_Y, parameters, _lambda, L)

            # Backward propagation
            if _lambda == 0 and keep_prob == 1:
                grads = backward_propagation(AL, minibatch_Y, caches)
            elif _lambda != 0:
                grads = backward_propagation_with_regularization(AL, minibatch_Y, caches, _lambda)
            elif keep_prob < 1:
                grads = backward_propagation_with_dropout(AL, minibatch_Y, caches, keep_prob)

            # Update parameters
            if optimizer == 'gd':
                parameters = update_parameters_using_gd(parameters, grads, learning_rate, L)
            elif optimizer == 'momentum':
                parameters, v = update_parameters_using_momentum(parameters, grads, v, beta, learning_rate, L)
            elif optimizer == 'adam':
                t += 1   # adam counter
                parameters, v, s = update_parameters_using_adam(parameters, grads, v, s, t, learning_rate, L, beta1, beta2, epsilon)

            cost_avg = cost_total / m

            # Print the cost every given epoch
            if print_cost and i % print_cost_after == 0:
                print ("Cost after epoch %i: %f" %(i, cost_avg))
            if print_cost and i % 100 == 0:
                costs.append(cost_avg)

        # Gradient checking
        gradient_checking(parameters, grads, X, Y, layers_dims, _lambda=_lambda) 

        return parameters