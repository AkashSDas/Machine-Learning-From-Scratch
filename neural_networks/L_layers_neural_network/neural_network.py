# L layers neural network


"""
    *** Neural Network Structure ***

    layers     : input - [L-1] hidden - output
    activations:         [L-1] Relu   - sigmoid

    cost function: 
        logprobs = np.multiply(-np.log(AL),Y) + np.multiply(-np.log(1 - AL), 1 - Y)
        cost = 1./m * np.nansum(logprobs)
"""


import numpy as np


# *** Initialize parameters ***

def initialize_parameters(layers_dims, seed=None):
    # layer_dims -- python array (list) containing the dimensions of each layer in our network

    if seed:
        np.random.seed(seed)

    parameters = {}
    L = len(layers_dims) # number of layers

    # Initializing parameters
    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layers_dims[l], layers_dims[l-1]) / np.sqrt(layers_dims[l-1]) # to have numerical stability dividing by np.sqrt(layers_dims[l-1])
        parameters[f'b{l}'] = np.zeros((layers_dims[l], 1))

    return parameters

# *** # Initialize parameters ***


# *** Activation functions ***

# Sigmoid function
def sigmoid(Z):
    A = 1/(1 + np.exp(-Z))
    cache = Z
    return A, cache

# Relu
def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

# *** # Activation functions ***


# *** Gradients of activation functions ***

""" 
    Used in backpropagation 

    dA -- post-activation gradient, of any shape
    dZ -- Gradient of the cost with respect to Z
"""

# Sigmoid
def sigmoid_backward(dA, cache):
    Z = cache
    S = 1/(1 + np.exp(-Z))
    dZ = dA * S * (1 - S)
    return dZ

# Relu
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

# *** # Gradients of activation functions ***


# *** Forward propagation ***

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

# *** # Forward propagation ***


# *** Compute cost ***

def compute_cost(AL, Y):
    m = Y.shape[1] # number of examples

    logprobs = np.multiply(-np.log(AL),Y) + np.multiply(-np.log(1 - AL), 1 - Y)
    cost = 1./m * np.nansum(logprobs)

    cost = np.squeeze(cost) # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    return cost

# *** # Compute cost ***


# *** Backward propagation ***

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

# *** # Backward propagation ***


# *** Update parameters ***

def update_parameters(parameters, grads, learning_rate, num_of_layers):
    L = num_of_layers # number of layers

    # Updating parameters
    for l in range(1, L+1):
        parameters[f'W{l}'] = parameters[f'W{l}'] - learning_rate * grads[f'dW{l}']
        parameters[f'b{l}'] = parameters[f'b{l}'] - learning_rate * grads[f'db{l}']

    return parameters

# *** # Update parameters ***


# *** Neural Network Model ***

def model(X, Y, layers_dims, learning_rate=0.01, num_of_iterations=10000, print_cost=True, print_cost_after=1000, seed=None):
    L = len(layers_dims) - 1 # number of layers

    # Initialize parameters
    parameters = initialize_parameters(layers_dims, seed)

    # Gradient Descent
    for i in range(num_of_iterations):
        # Forward propagation
        AL, caches = forward_propagation(X, parameters, L)

        # Compute cost
        cost = compute_cost(AL, Y)

        # Backward propagation
        grads = backward_propagation(AL, Y, caches)

        # Updating parameters
        parameters = update_parameters(parameters, grads, learning_rate, L)

        # Priniting cost after given iterations
        if print_cost and i % print_cost_after == 0:
            print ("Cost after iteration %i: %f" % (i, cost))

    return parameters

# *** # Neural Network Model ***


# Predict classes for given examples
def predict(parameters, X, num_of_layers, threshold=0.5):
    L = num_of_layers
    AL, caches = forward_propagation(X, parameters, L)
    AL[AL < threshold] = 0
    AL[AL >= threshold] = 1
    return AL # prediction


# Predict accuracy
def predict_accuracy(X, Y, parameters, num_of_layers, threshold=0.5):
    m = X.shape[1] # number of examples
    L = num_of_layers
    AL = predict(parameters, X, L, threshold)
    accuracy = np.sum((AL == Y) / m)
    return accuracy