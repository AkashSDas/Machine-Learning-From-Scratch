import numpy as np

from activations import sigmoid, relu, sigmoid_backward, relu_backward
from forward_propagation import linear_forward
from backpropagation import linear_backward


# ==================================================== 
# ### DROPOUT CHANGES ###
# ==================================================== 


# *** Forward propagation ***

# Applying dropout for `RELU` function only

def linear_activation_forward_with_dropout(A_prev, W, b, activation, keep_prob=0.5):
    # Linear forward step
    Z, linear_cache = linear_forward(A_prev, W, b)

    # Activation forward step
    if activation == 'relu':
        A, activation_cache = relu(Z)

        # Implementing dropout
        D = np.random.rand(A.shape[0], A.shape[1])
        D = (D < keep_prob).astype(int) # convert entries of D to 0 or 1 (using keep_prob as the threshold)
        A = A * D # shut down some neurons of A
        A = np.divide(A, keep_prob) # scale the value of neurons that haven't been shut down

        cache = (linear_cache, activation_cache, D)
    elif activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
        cache = (linear_cache, activation_cache, None)

    return A, cache


def forward_propagation_with_dropout(X, parameters, num_of_layers, keep_prob=0.5):
    caches = []
    L = num_of_layers 
    A = X

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward_with_dropout(A_prev, parameters[f'W{l}'], parameters[f'b{l}'], 'relu', keep_prob)
        caches.append(cache)

     # Implement LINEAR -> SIGMOID for ouput layer. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward_with_dropout(A, parameters[f'W{L}'], parameters[f'b{L}'], 'sigmoid', keep_prob)
    caches.append(cache)

    return AL, caches

# *** # Forward propagation ***


# *** Backward propagation ***

def linear_backward_with_dropout(dZ, cache, D, keep_prob):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward_with_dropout(dA, cache, activation, keep_prob):
    # Retriving cache
    linear_cache, activation_cache, D = cache

    # Linear backward and activation steps
    if activation == 'relu':
        # Implementing dropout
        dA = dA * D # Apply mask D to shut down the same neurons as during the forward propagation
        dA = np.divide(dA, keep_prob)  # Scale the value of neurons that haven't been shut down

        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward_with_dropout(dZ, linear_cache, D, keep_prob)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def backward_propagation_with_dropout(AL, Y, caches, keep_prob=0.5):
    grads = {}
    L = len(caches) # number of layers
    m = AL.shape[1] # number of layers
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    # Initializing backpropagation
    dAL = - np.divide(Y, AL) + np.divide(1-Y, 1-AL)

    # Lth layer (SIGMOID -> LINEAR) gradients
    current_cache = caches[L - 1]
    grads[f'dA{L-1}'], grads[f'dW{L}'], grads[f'db{L}'] = linear_activation_backward_with_dropout(dAL, current_cache, 'sigmoid', keep_prob)

    # Loop from L=L-1 to L=1
    for l in reversed(range(1, L)):
        # From L=L-1 to L=1: (RELU -> LINEAR) gradients

        current_cache = caches[l - 1]
        dA_prev_tmp, dW_tmp, db_tmp = linear_activation_backward_with_dropout(grads[f'dA{l}'], current_cache, 'relu', keep_prob)

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
    #     dA_prev_temp, dW_temp, db_temp = linear_activation_backward_with_dropout(grads[f'dA{l + 1}'], current_cache, 'relu')
                                                                                                  
    #     grads[f'dA{l}'] = dA_prev_temp
    #     grads[f'dW{l + 1}'] = dW_temp
    #     grads[f'db{l + 1}'] = db_temp

    return grads

# *** # Backward propagation ***
