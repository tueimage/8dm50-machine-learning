import numpy as np

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