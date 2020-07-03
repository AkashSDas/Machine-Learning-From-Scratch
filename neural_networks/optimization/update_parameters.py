import numpy as np


# Update parameters using gradient descent
def update_parameters_using_gd(parameters, grads, learning_rate, num_of_layers):
    L = num_of_layers # number of layers

    for l in range(1, L + 1):
        parameters[f'W{l}'] = parameters[f'W{l}'] - learning_rate * grads[f'dW{l}']  
        parameters[f'b{l}'] = parameters[f'b{l}'] - learning_rate * grads[f'db{l}']  

    return parameters


# ===========================================
# ### Momentum ###
# ===========================================

def initialize_velocity(parameters, num_of_layers):
    L = num_of_layers
    v = {}

    for l in range(1, L+1):
        v[f'dW{l}'] = np.zeros((parameters[f'W{l}'].shape))
        v[f'db{l}'] = np.zeros((parameters[f'b{l}'].shape))

    return v


# This update is without using bais correction
def update_parameters_using_momentum(parameters, grads, v, beta, learning_rate, num_of_layers):
    L = num_of_layers

    for l in range(1, L + 1):
        # Compute velocities
        v[f'dW{l}'] = (beta * v[f'dW{l}']) + ((1 - beta) * grads[f'dW{l}'])
        v[f'db{l}'] = (beta * v[f'db{l}']) + ((1 - beta) * grads[f'db{l}'])

        # Update parameters
        parameters[f'W{l}'] = parameters[f'W{l}'] - learning_rate * v[f'dW{l}']
        parameters[f'b{l}'] = parameters[f'b{l}'] - learning_rate * v[f'db{l}']

    return parameters, v


# ===========================================
# ### Adam ###
# ===========================================

def initialize_adam(parameters, num_of_layers):
    L = num_of_layers
    v = {} # exponential weighted average of the gradients
    s = {} # exponential weighted average of the squared gradients

    for l in range(1, L + 1):
        v[f'dW{l}'] = np.zeros((parameters[f'W{l}'].shape))
        v[f'db{l}'] = np.zeros((parameters[f'b{l}'].shape))
        s[f'dW{l}'] = np.zeros((parameters[f'W{l}'].shape))
        s[f'db{l}'] = np.zeros((parameters[f'b{l}'].shape))

    return v, s


def update_parameters_using_adam(parameters, grads, v, s, t, learning_rate, num_of_layers, beta1=0.9, beta2=0.999, epsilon=1e-8):
    # v -- Adam variable, moving average of the first gradient, python dictionary
    # s -- Adam variable, moving average of the squared gradient, python dictionary
    # beta1 -- Exponential decay hyperparameter for the first moment estimates 
    # beta2 -- Exponential decay hyperparameter for the second moment estimates 
    # epsilon -- hyperparameter preventing division by zero in Adam updates

    L = num_of_layers
    v_corrected = {} # initializing first moment estimate
    s_corrected = {} # initializing second moment estimate

    # Perform Adam updates for all parameters
    for l in range(1, L + 1):
        # Moving average of the gradients
        v[f'dW{l}'] = beta1 * v[f'dW{l}'] + (1-beta1) * grads[f'dW{l}']
        v[f'db{l}'] = beta1 * v[f'db{l}'] + (1-beta1) * grads[f'db{l}']

        # Compute bias-corrected first moment estimate
        v_corrected[f'dW{l}'] = v[f'dW{l}'] / (1 - np.power(beta1, t))
        v_corrected[f'db{l}'] = v[f'db{l}'] / (1 - np.power(beta1, t))

        # Moving average of the squared gradients
        s[f'dW{l}'] = beta2 * s[f'dW{l}'] + (1 - beta2) * np.power(grads[f'dW{l}'], 2)
        s[f'db{l}'] = beta2 * s[f'db{l}'] + (1 - beta2) * np.power(grads[f'db{l}'], 2)

        # Compute bias-corrected second raw moment estimate
        s_corrected[f'dW{l}'] = s[f'dW{l}'] / (1 - np.power(beta2, t))
        s_corrected[f'db{l}'] = s[f'db{l}'] / (1 - np.power(beta2, t))

        # Update parameters
        parameters[f'W{l}'] = parameters[f'W{l}'] - learning_rate * (v_corrected[f'dW{l}']/np.sqrt(s_corrected[f'dW{l}'] + epsilon))
        parameters[f'b{l}'] = parameters[f'b{l}'] - learning_rate * (v_corrected[f'db{l}']/np.sqrt(s_corrected[f'db{l}'] + epsilon))

    return parameters, v, s
