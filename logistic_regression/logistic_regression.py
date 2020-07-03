# Logistic regression


import numpy as np


# Mean normalization
def normalize(X):
    # Not normalizing 1's (constant term)
    X_ones = np.ones((len(X), 1)).reshape(1, -1)

    # Calculating mean along each column(axis=0)
    _mean = np.mean(X, axis=0)

    # Calculating standard deviation along each column(axis=0)
    _std = np.std(X, axis=0)

    # Calculating Z-score
    z_score = np.divide((X - _mean), _std)

    X = np.concatenate((X_ones.T, z_score), axis=1)

    return X


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Hypothesis
def hypothesis(X, theta):
    z = np.matmul(X, theta)
    return sigmoid(z)


# Predict
def predict(X, theta, threshold=0.5):
    # This is our probability vector
    prob_vect = hypothesis(X, theta)

    # Using list comprehension to check if probabilty >= threshold then 1 else 0
    result = [1 if p >= threshold else 0 for vect in prob_vect for p in vect]
    result = np.array(result).reshape(1, -1)
    return result


# Cost function
def cost(X, y, theta):
    # Lenght of our samples set, X
    m = len(X)
    h = hypothesis(X, theta)

    J = - (1 / m) * (np.matmul(y.T, np.log(h)) + np.matmul((1 - y).T, np.log(1 - h)))
    return J


# Gradient descent
def gradient_descent(X, y, theta, alpha, _lambda, num_of_iters):
    # Lenght of our samples set, X
    m = len(X)

    # Initializing column vector, J_history where all elements assigned to 0,
    # while iterating we will fill these values with those iterations cost function's value
    J_history = np.zeros((num_of_iters, 1))

    # Gradient Descent Algorithm Implementation
    for i in range(num_of_iters):
        h = hypothesis(X, theta)

        # Regularization parameter
        regularization_param = (_lambda / m) * theta

        # Regularizing all coefficients. This vectorized version of gradient descent
        tmp_1 = theta - (alpha * ((1 / m) * ((np.matmul(X.T, (h - y)) + regularization_param))))

        # We should NOT regularize the parameter theta_zero
        tmp_2 = theta[0] - (alpha * (1 / m) * (np.matmul((h - y).T, X[:, 0])))

        theta = tmp_1
        theta[0] = tmp_2

        J_history[i] = cost(X, y, theta)

    return (theta, J_history)