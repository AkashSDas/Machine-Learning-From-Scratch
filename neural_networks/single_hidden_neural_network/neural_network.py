# Single hidden layer neural network


"""
    *** Neural Network Structure ***

    layers     : input - hidden - output
    activations:         tanh     sigmoid
    

    cost function: 
        J = - (1/m) * np.sum(np.multiply(Y, np.log(A)) + np.multiply(1-Y, np.log(1-A)))
"""


import numpy as np


# Get layers sizes
def layers_sizes(X, Y):
    n_x = X.shape[0] # size of input layer
    n_y = Y.shape[0] # size of output layer

    return (n_x, n_y)


# Initialize parameters
def initialize_parameters(n_x, n_h, n_y, init_const=0.01, seed=None):
    if seed:
        np.random.seed(seed)

    # initializing parameters
    W1 = np.random.randn(n_h, n_x) * init_const
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * init_const
    b2 = np.zeros((n_y, 1))

    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return parameters


# Sigmoid function
def sigmoid(Z):
    S = 1/(1 + np.exp(-Z))
    return S


# Forward propagation
def forward_propagation(X, parameters):
    # Retriving parameters
    W1 = parameters['W1']         
    b1 = parameters['b1']         
    W2 = parameters['W2']         
    b2 = parameters['b2']

    # Implementing forward propagation
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    # Cache for efficient backward propagation
    cache = {
        'Z1': Z1,
        'A1': A1,
        'Z2': Z2,
        'A2': A2
    }

    return A2, cache


# Compute cost
def compute_cost(A2, Y, parameters):
    m = Y.shape[1] # number of examples

    # Retriving parameters
    W1 = parameters['W1']         
    W2 = parameters['W2']

    cost = - (1/m) * np.sum(np.multiply(Y, np.log(A2)) + np.multiply(1-Y, np.log(1-A2)))
    cost = np.squeeze(cost) # makes sure cost is the dimension we expect. # E.g., turns [[17]] into 17

    return cost


# Backward propagation
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1] # number of examples

    # Retriving parameters
    W1 = parameters['W1']         
    W2 = parameters['W2']

    # Retriving caches
    A1 = cache['A1']
    A2 = cache['A2']

    # Implementing backward propagation
    dZ2 = A2 - Y
    dW2 = (1/m) * (np.dot(dZ2, A1.T))
    db2 = (1/m) * (np.sum(dZ2, axis=1, keepdims=True))
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1/m) * (np.dot(dZ1, X.T))
    db1 = (1/m) * (np.sum(dZ1, axis=1, keepdims=True))

    # Gradients(derivatives)
    grads = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2
    }

    return grads


# Update parameters
def update_parameters(parameters, grads, learning_rate=0.01):
    # Retriving parameters
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    # Retriving gradient
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    # Updating parameters
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return parameters


# Neural Network model
def model(X, Y, n_h, learning_rate=0.01, init_const=0.01, num_of_iterations=10000, print_cost=True, print_cost_after=1000, seed=None):
    n_x, n_y = layers_sizes(X, Y)
    
    # Initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y, init_const, seed)

    # Gradient Descent
    for i in range(num_of_iterations):
        # Forward propagation
        A2, cache = forward_propagation(X, parameters)

        # Compute cost
        cost = compute_cost(A2, Y, parameters)

        # Backward propagation
        grads = backward_propagation(parameters, cache)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Priniting cost after given iterations
        if print_cost and i % print_cost_after == 0:
            print ("Cost after iteration %i: %f" % (i, cost))

    return parameters


# Predict classes for given examples
def predict(parameters, X, threshold=0.5):
    A2, cache = forward_propagation(X, parameters)
    A2[A2 < threshold] = 0
    A2[A2 >= threshold] = 1
    return A2 # predictions


# Predict classes for given examples
def predict(parameters, X, threshold=0.5):
    A2, cache = forward_propagation(X, parameters)
    A2[A2 < threshold] = 0
    A2[A2 >= threshold] = 1
    return A2 # predictions
