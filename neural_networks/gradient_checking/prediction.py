import numpy as np

from forward_propagation import forward_propagation


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
