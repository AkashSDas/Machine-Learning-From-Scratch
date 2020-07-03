import numpy as np


# Update parameters
def update_parameters(parameters, grads, learning_rate, num_of_layers):
    L = num_of_layers # number of layers

    # Updating parameters
    for l in range(1, L+1):
        parameters[f'W{l}'] = parameters[f'W{l}'] - learning_rate * grads[f'dW{l}']
        parameters[f'b{l}'] = parameters[f'b{l}'] - learning_rate * grads[f'db{l}']

    return parameters
