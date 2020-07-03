# Initialize parameters in different ways


"""
    *** Different Initialization Ways ***

    1. Zeros
    2. Random
    3. He initialization
    4. Xavier initialization
"""


import numpy as np


# *** Different methods of initializing parameters ***

# Zeros initialization
def initialize_parameters_zeros(layers_dims):
    parameters = {}
    L = len(layers_dims) # number of layers

    for l in range(1, L):
        parameters[f'W{l}'] = np.zeros((layers_dims[l], layers_dims[l-1]))
        parameters[f'b{l}'] = np.zeros((layers_dims[l], 1))

    return parameters

# Random initialization
def initialize_parameters_random(layers_dims, init_const=0.01, seed=None):
    if seed:
        np.random.seed(seed)

    parameters = {}
    L = len(layers_dims) # number of layers

    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layers_dims[l], layers_dims[l-1]) * init_const
        parameters[f'b{l}'] = np.zeros((layers_dims[l], 1))

    return parameters

# He initialization
def initialize_parameters_he(layers_dims, seed=None):
    # Use when you're using `RELU` as activation function
    if seed:
        np.random.seed(seed)

    parameters = {}
    L = len(layers_dims) # number of layers
    
    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1]) 
        parameters[f'b{l}'] = np.zeros((layers_dims[l], 1))
    
    return parameters

# Xavier initialization
def initialize_parameters_xavier(layers_dims, seed=None):
    # Use when you're using `tanh` as activation function

    if seed:
        np.random.seed(seed)

    parameters = {}
    L = len(layers_dims) # number of layers
    
    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(1/layers_dims[l-1]) 
        parameters[f'b{l}'] = np.zeros((layers_dims[l], 1))
    
    return parameters

# *** # Different methods of initializing parameters ***


# *** Initialize parameters ***

def initialize_parameters(layers_dims, initialization='random', init_const=0.01, seed=None):
    if initialization == 'random':
        return initialize_parameters_random(layers_dims, init_const, seed)
    elif initialization == 'he':
        return initialize_parameters_he(layers_dims, seed)
    elif initialization == 'xavier':
        return initialize_parameters_xavier(layers_dims, seed)
    elif initialization == 'zeros':
        return initialize_parameters_zeros(layers_dims)

# *** # Initialize parameters ***