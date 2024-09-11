import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def normalize_features(X_train, X_test):
    """
    Normalize the features using StandardScaler (z-score normalization).
    :param X_train: Training data features
    :param X_test: Test data features
    :return: Normalized X_train and X_test
    """
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    return X_train_normalized, X_test_normalized

def euclidean_distance(point_1, point_2):
    """
    Calculate the Euclidean distance between two points.
    :param point_1: First point (array-like)
    :param point_2: Second point (array-like)
    :return: Euclidean distance between the two points
    """
    return np.sqrt(np.sum((point_1 - point_2) ** 2, axis=1))

def knn_regression_predict(X_train, y_train, X_test_sample, k):
    """
    Predict target value for a single test sample using k-NN regression.
    :param X_train: Training data features
    :param y_train: Training data target values
    :param X_test_sample: Test sample for prediction
    :param k: Number of neighbors to use for prediction
    :return: Predicted target value for the test sample
    """
    # Calculate distances to all training points
    distances = euclidean_distance(X_train, X_test_sample)

    # Get indices of the k-nearest neighbors
    indices = np.argsort(distances)[:k]

    # Return the average of the k nearest neighbors' target values
    return np.mean(y_train[indices])

def knn_regression(X_train, y_train, X_test, k):
    """
    Predict target values for all test samples using k-NN regression.
    :param X_train: Training data features
    :param y_train: Training data target values
    :param X_test: Test data features
    :param k: Number of neighbors to use for prediction
    :return: Predicted target values for all test samples
    """
    predictions = [knn_regression_predict(X_train, y_train, sample, k) for sample in X_test]
    return np.array(predictions)
