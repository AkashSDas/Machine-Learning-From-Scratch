import numpy as np

from forward_propagation import forward_propagation
from compute_cost import compute_cost

from regularization import compute_cost_with_regularization
from dropout import forward_propagation_with_dropout


def dictionary_to_vector(parameters):
    for i, key in enumerate(parameters.keys()):
        new_vector = np.reshape(parameters[key], (-1, 1))
        if i == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        
    return theta


def vector_to_dictionary(theta, layers_dims):
    L = len(layers_dims)
    parameters = {}
    k = 0
    for l in range(1, L):
        # Create tmp variable to store dimension used on each layer
        w_dim = layers_dims[l] * layers_dims[l-1]
        b_dim = layers_dims[l]

        # Create tmp var to be used in slicing theta vector
        tmp_dim = k + w_dim
        
        # Add theta to the dictionary
        parameters[f'W{l}'] = theta[k:tmp_dim].reshape(layers_dims[l], layers_dims[l-1])
        parameters[f'b{l}'] = theta[tmp_dim:tmp_dim + b_dim].reshape(b_dim, 1)

        k += w_dim + b_dim
        
    return parameters


def gradients_to_vector(gradients, num_of_layers):
    L = num_of_layers
    keys = []
    for l in range(1, L):
        keys.append(f'dW{l}')
        keys.append(f'db{l}')

    filtered_gradients = {}
    for key in keys:
        filtered_gradients[key] = gradients[key]

    gradients = filtered_gradients

    for i, key in enumerate(gradients):
        new_vector = np.reshape(gradients[key], (-1, 1))
        if i == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        
    return theta


# Gradient checking
def gradient_checking(parameters, gradients, X, Y, layers_dims, _lambda=0, keep_prob=1, epsilon=1e-7):
    # Set-up variables
    parameters_values = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients, len(layers_dims))
    num_of_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_of_parameters, 1))
    J_minus = np.zeros((num_of_parameters, 1))
    grad_approx = np.zeros((num_of_parameters, 1))
    num_of_layers = len(layers_dims) - 1

    # Compute grad_approx
    for i in range(num_of_parameters):
        # Compute J_plus[i]
        theta_plus = np.copy(parameters_values)
        theta_plus[i][0] = theta_plus[i][0] + epsilon
        if keep_prob == 1:
            AL, _ = forward_propagation(X, vector_to_dictionary(theta_plus, layers_dims), num_of_layers)
        elif keep_prob < 1:
            AL, _ = forward_propagation_with_dropout(X, vector_to_dictionary(theta_plus, layers_dims), num_of_layers, keep_prob)
        if _lambda == 0:
            J_plus[i] = compute_cost(AL, Y)
        else:
            J_plus[i] = compute_cost_with_regularization(AL, Y, parameters, _lambda, num_of_layers)

        # Compute J_minus[i]
        theta_minus = np.copy(parameters_values)
        theta_minus[i][0] = theta_minus[i][0] - epsilon
        if keep_prob == 1:
            AL, _ = forward_propagation(X, vector_to_dictionary(theta_minus, layers_dims), num_of_layers)
        elif keep_prob < 1:
            AL, _ = forward_propagation_with_dropout(X, vector_to_dictionary(theta_minus, layers_dims), num_of_layers, keep_prob)
        if _lambda == 0:
            J_minus[i] = compute_cost(AL, Y)
        else:
            J_minus[i] = compute_cost_with_regularization(AL, Y, parameters, _lambda, num_of_layers)
        
        # Compute grad_approx[i]
        grad_approx[i] = np.divide(J_plus[i] - J_minus[i], 2 * epsilon)

    # Compare gradapprox to backward propagation gradients by computing difference
    numerator = np.linalg.norm(grad - grad_approx)
    denominator = np.linalg.norm(grad) + np.linalg.norm(grad_approx)
    difference = np.divide(numerator, denominator)

    if difference > 2e-7:
        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference
