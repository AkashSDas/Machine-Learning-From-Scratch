import numpy as np


# Compute cost
def compute_cost(AL, Y):
    m = Y.shape[1] # number of examples

    logprobs = np.multiply(-np.log(AL),Y) + np.multiply(-np.log(1 - AL), 1 - Y)
    cost = 1./m * np.nansum(logprobs)

    cost = np.squeeze(cost) # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    return cost
