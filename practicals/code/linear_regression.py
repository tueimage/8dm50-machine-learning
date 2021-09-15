import numpy as np

def lsq(X, y):
    """
    Least squares linear regression
    :param X: Input data matrix
    :param y: Target vector
    :return: Estimated coefficient vector for the linear regression
    """

    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # calculate the coefficients
    beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return beta

def mse(X, y, beta):
    """
    Mean squared error of linear regression model
    :param X:   Input data matrix
    :param y:   Target vector
    :param beta: Estimated coefficient vector for the linear regression
    :return:    Mean squared error
    """

    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)
    
    # compute the error
    epsilon = np.transpose(X.dot(beta) - y).dot(X.dot(beta) - y) / len(y)
    
    return epsilon