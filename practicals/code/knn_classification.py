import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode

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
    return np.sqrt(np.sum((point_1 - point_2)**2, axis=1))

def knn_predict(X_train, y_train, X_test_sample, k):
    """
    Predict class label for a single test sample using k-NN classification.
    :param X_train: Training data features
    :param y_train: Training data labels
    :param X_test_sample: Test sample for prediction
    :param k: Number of neighbors to use for prediction
    :return: Predicted class label for the test sample
    """
    # Calculate distances to all training points
    distances = euclidean_distance(X_train, X_test_sample)

    # Get indices of the k-nearest neighbors
    indices = np.argsort(distances)[:k]

    # Get labels of the k-nearest neighbors
    labels = y_train[indices]

    # Return the most common label (mode) among the neighbors
    return mode(labels).mode[0]

def knn_classification(X_train, y_train, X_test, k):
    """
    Predict class labels for all test samples using k-NN classification.
    :param X_train: Training data features
    :param y_train: Training data labels
    :param X_test: Test data features
    :param k: Number of neighbors to use for prediction
    :return: Predicted class labels for all test samples
    """
    predictions = [knn_predict(X_train, y_train, sample, k) for sample in X_test]
    return np.array(predictions)

def get_accuracy(y_true, y_pred):
    """
    Compute accuracy of the k-NN classification predictions.
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Accuracy of the predictions
    """
    return np.sum(y_true == y_pred) / len(y_true)
