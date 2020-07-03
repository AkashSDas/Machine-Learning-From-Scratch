import numpy as np

from activations import sigmoid_backward, relu_backward


"""
    Backward propagation structure
        input layer -> [Linear -> Relu] * (L - 1) -> [Linear -> Sigmoid] (output layer)

    One unit does two work first calculate Z (linear forward) and then calculate A (activation)
"""


# Linear backward step
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1] # number of examples

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


# Combining linear backward and activation steps
def linear_activation_backward(dA, cache, activation):
    # Retriving cache
    linear_cache, activation_cache = cache

    # Backward activation step
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)

    # Linear backward step
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


# Backward propagation for L layers
def backward_propagation(AL, Y, caches):
    grads = {}
    L = len(caches) # number of layers
    m = AL.shape[1] # number of examples
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    # Initializing backpropagation
    dAL = - np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)

    # Lth layer (SIGMOID -> LINEAR) gradients
    current_cache = caches[L - 1]
    grads[f'dA{L-1}'], grads[f'dW{L}'], grads[f'db{L}'] = linear_activation_backward(dAL, current_cache, 'sigmoid')

    # Loop from L=L-1 to L=1
    for l in reversed(range(1, L)):
        # lth layer: (RELU -> LINEAR) gradients.

        current_cache = caches[l - 1]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads[f'dA{l}'], current_cache, 'relu')
                                                                                                  
        grads[f'dA{l - 1}'] = dA_prev_temp
        grads[f'dW{l}'] = dW_temp
        grads[f'db{l}'] = db_temp

    """
        Below commented loop is another way to implement above loop
        but it is made simpler to understand what's going on.
    """

    # # Loop from L=L-2 to L=0
    # for l in reversed(range(L - 1)):
    #     # lth layer: (RELU -> LINEAR) gradients.

    #     current_cache = caches[l]
    #     dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads[f'dA{l + 1}'], current_cache, 'relu')
                                                                                                  
    #     grads[f'dA{l}'] = dA_prev_temp
    #     grads[f'dW{l + 1}'] = dW_temp
    #     grads[f'db{l + 1}'] = db_temp

    return grads
