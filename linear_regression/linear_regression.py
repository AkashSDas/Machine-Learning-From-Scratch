# Linear Regression 


import numpy as np


# Mean normalisation
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


# Hypothesis
def hypothesis(X, theta):
    return np.matmul(X, theta)


# Cost function
def compute_cost(X, y, theta):
    # Lenght of our samples set, X
    m = len(X)
    h = hypothesis(X, theta)

    # mean-square-error
    sqr_error = (h - y) ** 2

    # Cost
    J = (1 / (2 * m)) * sum(sqr_error)[0]
    return J

    # We are summing the mean-square-errors by using the sum object,
    # the sum object returns a list of single element that is our
    # sum of mean-square-errors and taking the value from 0th index


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
        regularization_param = 1 - alpha * _lambda / m

        # Regularizing all coefficients. This vectorized version of gradient descent
        tmp_1 = theta * regularization_param - ((alpha / m) * (np.matmul((h - y).T, X)).T)

        # We should NOT regularize the parameter theta_zero
        tmp_2 = theta[0] - ((alpha / m) * (np.matmul((h - y).T, X[:, 0])).T)

        theta = tmp_1
        theta[0] = tmp_2

        J_history[i] = compute_cost(X, y, theta)

    return (theta, J_history)


# Normal equation
def normal_equation(X, y, _lambda):
    # Number of rows
    n = len(X[0])

    # Creating a identity matrix whose 0th row's 0th element is 0
    L = np.identity(n)
    L[0][0] = 0

    # Regularization parameter
    regularization_param = _lambda * L

    # Matrix multiplication of X transpose and y
    a = np.matmul(X.T, y)

    # Matrix multiplication of X transpose and X
    b = np.matmul(X.T, X)

    # Adding the regularization parameter
    c = b + regularization_param

    # Taking inverse of matrix b
    d = np.linalg.pinv(c)

    # Matrix multiplication of c transpose and a
    theta = np.matmul(d, a)

    return theta
