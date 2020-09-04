import numpy as np

def mse(y_predict, y_true):
    """
    function to evaluate the mean squared error

    :params y_predict: numpy vector with the predicted value
    :params y_true: numpy vector with the true value
    :returns mse: mean squared error of the prediction

    """
    if y_predict.shape != y_true.shape:
        print('The vectors with the prediction and the true value should be of the same size')
        return

    error = y_true - y_predict
    squared_error = error*error
    mean_squared_error = np.sum(squared_error) / squared_error.shape[0]
    
    return mean_squared_error

