import numpy as np
import scipy as sp

def euclidean_distance(a, b):
    """calculate euclidean distance between two points a and b"""
    return np.linalg.norm(a-b)

def get_neighbours(X_train, X_test, k):
    """finds the k amount closest neighbours and 
    returns the indeces and distances of the closest neigbours
    
    parameters
    ----------
    X_train : x train set
    X_test : x test set
    k : amount of neigbours"""
    distances = np.zeros(len(X_train))
    for i in range(len(X_train)):
        distances[i] = euclidean_distance(X_test, X_train[i])
    index = np.argsort(distances)
    distances = np.sort(distances)
    return index[:k], distances[:k]

def kNN_classification(X_train, X_test, y_train, k):
    """calculate the prediction of the y values using kNN classification
    
    parameters
    ----------
    X_train : x train set
    X_test : x test set
    y_train : y train set
    k : amount of neighbours"""
    y_pred = np.zeros(len(X_test))
    for i, x_test in enumerate(X_test):
        index, dist = get_neighbours(X_train, x_test, k)
        if np.sum(y_train[index,0]) > k/2:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return y_pred[:, np.newaxis]

def kNN_regression(X_train, y_train, X_test, k):
    """calculate the prediction of the y values using kNN regression
    
    parameters
    ----------
    X_train : x train set
    X_test : x test set
    y_train : y train set
    k : amount of neighbours"""
    y_pred = np.zeros(len(X_test), dtype=int)
    for i, x_test in enumerate(X_test):
        index, dist = get_neighbours(X_train, x_test, k)
        y_pred[i] = np.sum(y_train[index,0])/k
    return y_pred[:, np.newaxis]

def get_neighbour_targets_numpy(X_train, X_test, y_train, k):
    """
    get the target values from the k nearest neighbors

    Parameters
    ----------
    X_train : numpy.ndarray
        numpy array containing training samples
    X_test : numpy.ndarray
        numpy array containing test samples
    y_train : numpy.ndarray
        numpy array containing training targets
    k : int
        number of neighbors to account for (preferably odd)

    Returns
    -------
    y_predic : numpy.ndarray
        numpy array containing predicted targets

    """
    # Create 3D matrix where the second dimension is the number of 
    # samples in X and the third dimension is the number of features
    # this has to be like this because now the last two axes of X_three
    # and X are the same and can be subtracted easily
    X_three = np.repeat(X_train[:, :, None], len(X_test), axis=2)
    X_three = np.swapaxes(X_three, 1, 2)
    
    # Subtract features of samples from 3D matrix
    X_subtract = X_three - X_test
    
    # Square, sum, square root, to get L2 norm
    X_l2 = np.sqrt(np.sum(np.multiply(X_subtract, X_subtract), axis = 2))
    
    # Get indices of the smallest
    # mind that argpartition does not sort, instead use np.argsort if wanted
    X_idx_k = np.argpartition(X_l2, k, axis=0)[:k, :]
    target_size = X_idx_k.shape
    
    # Get values and reshape
    y_k = y_train[X_idx_k.flatten()]
    y_k = y_k.reshape(target_size)
    
    return y_k


def kNN_classification_numpy(X_train, X_test, y_train, k):
    """
    Classify X_test using the k nearest neighbors and the training data
    X_train and y_train

    Parameters
    ----------
    X_train : numpy.ndarray
        numpy array containing training samples
    X_test : numpy.ndarray
        numpy array containing test samples
    y_train : numpy.ndarray
        numpy array containing training targets
    k : int
        number of neighbors to account for (preferably odd)

    Returns
    -------
    y_predic : numpy.ndarray
        numpy array containing predicted targets

    """
    y_k = get_neighbour_targets_numpy(X_train, X_test, y_train, k)
    
    return np.swapaxes(sp.stats.mode(y_k)[0], 0, 1)

def kNN_regression_numpy(X_train, X_test, y_train, k):
    """
    Regress X_test using the k nearest neighbors and the training data
    X_train and y_train

    Parameters
    ----------
    X_train : numpy.ndarray
        numpy array containing training samples
    X_test : numpy.ndarray
        numpy array containing test samples
    y_train : numpy.ndarray
        numpy array containing training targets
    k : int
        number of neighbors to account for (preferably odd)

    Returns
    -------
    y_predic : numpy.ndarray
        numpy array containing predicted targets

    """
    y_k = get_neighbour_targets_numpy(X_train, X_test, y_train, k)
    
    return np.mean(y_k, axis=0)[:, None]

def kNN_find_optimal_k(X_train, X_test, y_train, kNN_type, use_numpy, k_array):
    targets = []
    
    if kNN_type == 'classification':
        if use_numpy == True:
            for k in k_array:
                targets.append(kNN_classification_numpy(
                    X_train,
                    X_test,
                    y_train,
                    k
                ))
        else:
            for k in k_array:
                targets.append(kNN_classification(
                    X_train, 
                    X_test, 
                    y_train, 
                    k
                ))
                
    elif kNN_type == 'regression':
        if use_numpy == True:
            for k in k_array:
                targets.append(kNN_regression_numpy(
                    X_train,
                    X_test,
                    y_train,
                    k
                ))
        else:
            for k in k_array:
                targets.append(kNN_regression(
                    X_train, 
                    X_test, 
                    y_train, 
                    k
                ))
    else: print("kNN_type should be one of 'classification' or 'regression'")
    
    return np.asarray(targets)
                