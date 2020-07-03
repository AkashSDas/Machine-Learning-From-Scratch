import numpy as np

from activations import sigmoid, relu


"""
    Forward propagation structure:
        input layer -> [Linear -> Relu] * (L - 1) -> [Linear -> Sigmoid] (output layer)

    One unit does two work first calculate Z (linear forward) and then calculate A (activation)
"""


# linear forward
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


# Combining linear forward and activation steps
def linear_activation_forward(A_prev, W, b, activation):
    # Linear forward step
    Z, linear_cache = linear_forward(A_prev, W, b)

    # Forward activation step
    if activation == 'relu':
        A, activation_cache = relu(Z)
    elif activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


# Forward propagation for L layers
def forward_propagation(X, parameters, num_of_layers):
    caches = []
    A = X
    L = num_of_layers

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters[f'W{l}'], parameters[f'b{l}'], 'relu')
        caches.append(cache)

    # Implement LINEAR -> SIGMOID for ouput layer. Add "cache" to the "caches" list
    AL, cache = linear_activation_forward(A, parameters[f'W{L}'], parameters[f'b{L}'], 'sigmoid')
    caches.append(cache)

    return AL, caches
