import math
import numpy as np

from initialization import initialize_parameters
from forward_propagation import forward_propagation
from compute_cost import compute_cost
from backpropagation import backward_propagation

from regularization import compute_cost_with_regularization, backward_propagation_with_regularization
from dropout import forward_propagation_with_dropout, backward_propagation_with_dropout
from gradient_checking import gradient_checking

from update_parameters import update_parameters_using_gd


# ===========================================
# ### Mini-Batch Gradient Descent ###
# ===========================================


# Mini-Batch Gradient Descent
def random_mini_batches(X, Y, mini_batch_size=64, seed=None):
    if seed:
        np.random.seed(seed)

    m = X.shape[1] # number of training examples
    mini_batches = []

    # Shuffle (X, Y)
    permutations = list(np.random.permutation(m))
    shuffled_X = X[:, permutations]
    shuffled_Y = Y[:, permutations].reshape((1, m))

    # Partition (shuffled_X, shuffles_Y), 
    # minus the last case where number of examples might be different
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(num_complete_minibatches):
        _from = k * mini_batch_size
        _to = (k + 1) * mini_batch_size

        mini_batch_X = shuffled_X[:, _from:_to]
        mini_batch_Y = shuffled_Y[:, _from:_to]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        _from = num_complete_minibatches * mini_batch_size
        mini_batch_X = shuffled_X[:, _from:]
        mini_batch_Y = shuffled_Y[:, _from:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches
