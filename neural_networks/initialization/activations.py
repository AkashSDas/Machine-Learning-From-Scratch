import numpy as np


# Sigmoid function
def sigmoid(Z):
    A = 1/(1 + np.exp(-Z))
    cache = Z
    return A, cache


# Relu function
def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


""" 
    Used in backpropagation 

    dA -- post-activation gradient, of any shape
    dZ -- Gradient of the cost with respect to Z
"""


# Gradient of Sigmoid function
def sigmoid_backward(dA, cache):
    Z = cache
    S = 1/(1 + np.exp(-Z))
    dZ = dA * S * (1 - S)
    return dZ


# Gradient of Relu function
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ
