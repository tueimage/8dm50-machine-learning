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

def predict(X, beta):
    """
    Prediction of the target variable using the estimated coefficients
    :param X: Input data matrix
    :param beta: Estimated coefficients
    :return: Predicted target variable
    """
    # Add a column of ones to calculate the intercept 
    ones = np.ones((len(X), 1))

    # Append column
    X_with_ones = np.concatenate((ones, X), axis=1)
    
    return np.dot(X_with_ones, beta)

def mse_calc(y_true, y_predicted):
    """
    Mean squared error
    :param y_true: True target values
    :param y_pred: Predicted target values
    :return: Mean squared error
    """
    return np.mean((y_true - y_predicted) ** 2)


def wlsq(X, y, w):
    """
    Weighted least squares linear regression
    :param X: Input data matrix
    :param y: Target vector
    :param w: Weights vector
    :return: Estimated coefficient vector for the linear regression
    """

    # Add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    W = np.diag(w)

    XT_W = np.dot(X.T, W)
    beta = np.dot(np.linalg.inv(np.dot(XT_W, X)), np.dot(XT_W, y))

    return beta