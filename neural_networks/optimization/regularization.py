import numpy as np

from activations import sigmoid_backward, relu_backward
from compute_cost import compute_cost


# ==================================================== 
# ### REGULARIZATION CHANGES ###
# ==================================================== 


# Cost function
def compute_cost_with_regularization(AL, Y, parameters, _lambda, num_of_layers):
    m = Y.shape[1] # number of examples

    # Compute sum of squares of parameters
    W = 0
    for i in range(1, num_of_layers + 1):
        W += np.sum(np.square(parameters[f'W{i}']))

    # Regularization parameters
    L2_regularization_cost = (1/m) * (_lambda/2) * W

    # Cross entropy cost
    cross_entropy_cost = compute_cost(AL, Y)

    cost = cross_entropy_cost + L2_regularization_cost
    return cost


# *** Backpropagation ***

# changes will come only in the way we are computing linear backward
def linear_backward_with_regularization(dZ, cache, _lambda):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T) + ((_lambda/m) * W)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db 


def linear_activation_backward_with_regularization(dA, cache, activation, _lambda):
    # Retriving cache
    linear_cache, activation_cache = cache

    # Activation backward step
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
    
    # Linear backward step
    dA_prev, dW, db = linear_backward_with_regularization(dZ, linear_cache, _lambda)

    return dA_prev, dW, db


def backward_propagation_with_regularization(AL, Y, caches, _lambda):
    grads = {}
    L = len(caches) # number of layers
    m = AL.shape[1] # number of examples
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    # Initializing backpropagation
    dAL = - np.divide(Y, AL) + np.divide(1-Y, 1-AL)

    # Lth layer (SIGMOID -> LINEAR) gradients
    current_cache = caches[L - 1]
    grads[f'dA{L-1}'], grads[f'dW{L}'], grads[f'db{L}'] = linear_activation_backward_with_regularization(dAL, current_cache, 'sigmoid', _lambda)

    # Loop from L=L-1 to L=1
    for l in reversed(range(1, L)):
        # From L=L-1 to L=1: (RELU -> LINEAR) gradients

        current_cache = caches[l - 1]
        dA_prev_tmp, dW_tmp, db_tmp = linear_activation_backward_with_regularization(grads[f'dA{l}'], current_cache, 'relu', _lambda)

        grads[f'dA{l - 1}'] = dA_prev_tmp
        grads[f'dW{l}'] = dW_tmp
        grads[f'db{l}'] = db_tmp

    """
        Below commented loop is another way to implement above loop
        but it is made simpler to understand what's going on.
    """

    # # Loop from L=L-2 to L=0
    # for l in reversed(range(L - 1)):
    #     # lth layer: (RELU -> LINEAR) gradients.

    #     current_cache = caches[l]
    #     dA_prev_temp, dW_temp, db_temp = linear_activation_backward_with_regularization(grads[f'dA{l + 1}'], current_cache, 'relu')
                                                                                                  
    #     grads[f'dA{l}'] = dA_prev_temp
    #     grads[f'dW{l + 1}'] = dW_temp
    #     grads[f'db{l + 1}'] = db_temp

    return grads

# *** # Backpropagation ***