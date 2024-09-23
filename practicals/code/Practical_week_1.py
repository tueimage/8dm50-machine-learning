
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:12:26 2024

@author: 20201969
"""
import numpy as np
from scipy import stats as st

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

"""
k-NN classification
"""

def normalize(dataset):
    """
    Normalize a dataset using Min-Max scaling.
    Each element in the dataset is scaled to a range between 0 and 1. 
    :param dataset: Input data matrix, where rows represent samples and columns represent features
    :return: Normalized data matrix where each feature value lies between 0 and 1  
    """
    # Compute the minimum and maximum values of the dataset
    x_min= np.min(dataset)
    x_max= np.max(dataset)
    
    # Apply the Min-Max normalization formula
    dataset_norm=(dataset-x_min)/(x_max-x_min)
    
    return dataset_norm 

def find_closests(X_train, X_test, k):
    """
    Finds the indices of the k nearest neighbors in X_train for each sample in X_test.
    :param X_train: Matrix of training data where each row is a training example and 
                    columns represent features.
    :param X_test: Matrix of test data where each row is a test example and columns 
                   represent features.
    :param k: Number of nearest neighbors to return for each test sample.  
    :return: Maxtrix containing the indices of the k nearest neighbors from X_train 
             for eachsample in X_test. 
    """
    # Calculate the sum of the squares of each element per row
    X_train_squared_sum = np.sum(X_train**2, axis = 1) 
    X_test_squared_sum = np.sum(X_test**2, axis = 1)

    # Compute the product of X_test and the transpose of X_train
    X_multiplied = np.matmul(X_test, X_train.T) 

    # Compute the pairwise Euclidean distances
    X_test_squared_sum = X_test_squared_sum.reshape(-1,1) 
    dists = np.sqrt(X_test_squared_sum - 2*X_multiplied + X_train_squared_sum)

    # Determine the indices of the k smallest distances (nearest neighbors)
    minimum_indices = np.argpartition(dists, kth = k-1, axis = -1)[:,:k]
    
    return minimum_indices

def pred_Y(minimum_indices, y_train):
     """
    Predict each test sample's most frequent class label based on its k nearest neighborss.
    :param minimum_indices: Indices of the k nearest neighbors for each test sample
    :param y_train: Class labels of the training data  
    :return: Predicted class labels
     """
     # Get the labels of the k nearest neighbors for each test sample
     nearest_neighbors_outcomes = y_train[[minimum_indices]].reshape(minimum_indices.shape)
     
     # Find the most frequent label (mode) among the k neigh 
     outcome = st.mode(np.transpose(nearest_neighbors_outcomes))[0]
     
     return outcome

def calculate_accuracy(pred, real):
    """
    Calculate the accuracy of predictions compared to the true labels.
   
    :param pred: Predicted labels 
    :param real: True labels
    :return: Accuracy as a float
   """
    
    # Ensure both arrays have the same shape
    real = real.reshape(pred.shape)

    # Count the number of errors squared (non-matching predictions)
    errors = np.sum((real-pred)**2)

    # Calculate accuracy: number of good outcomes devided by total number of outcomes
    accuracy = (real.size-errors)/real.size
    
    return accuracy
    
def knnClassifier(X_train, X_test, y_train, y_test, k):
    """
    Perform k-Nearest Neighbors classification and return the accuracy.
   
    :param X_train: Training data (features)
    :param X_test: Test data (features)
    :param y_train: Training data labels
    :param y_test: Test data labels
    :param k: Number of nearest neighbors
    :return: Accuracy of the classification
    """
    # Normalize training and test data
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    
    # Find the k nearest neighbors for each test sample
    minimum_indices = find_closests(X_train, X_test, k)

    # Predict the labels for the test set
    pred = pred_Y(minimum_indices, y_train)
    
    # Calculate and return the accuracy
    acc = calculate_accuracy(pred, y_test)
    
    return acc
    
"""
k-NN regression
"""

def regression_Y(minimum_indices, y_train):
    """
    Predict the output for each test sample based on the mean of the k nearest neighbors' outcomes.
    
    :param minimum_indices: Indices of the k nearest neighbors for each test sample (shape: n_test_samples, k)
    :param y_train: Target values of the training data (shape: n_train_samples,)
    :return: Predicted values for each test sample (shape: n_test_samples,)
    """
    
    # Get the target values of the k nearest neighbors
    nearest_neighbors_outcomes = y_train[[minimum_indices]].reshape(minimum_indices.shape)

    # Calculate the mean of the nearest neighbor outcomes
    outcome = np.mean(nearest_neighbors_outcomes, axis=1)
    
    return outcome
    
def calculate_regr_std(pred, real):
    """
    Calculate the standard deviation of the residuals for regression predictions.
    
    :param pred: Predicted values 
    :param real: True values 
    :return: Standard deviation of the residuals
    """
    # Ensure both arrays have the same shape
    real = real.reshape(pred.shape)
    
    # Calculate the squared differences
    errors = sum((real-pred)**2)
    
    # Calculate and return the standard deviation
    std = (errors/real.size)**0.5
    
    return std

def knnRegressor(X_train, X_test, y_train, y_test, k):
    """
    Perform k-Nearest Neighbors regression and return the standard deviation of the residuals.
    
    :param X_train: Training data (features)
    :param X_test: Test data (features)
    :param y_train: Training data labels (target values)
    :param y_test: Test data labels (true values)
    :param k: Number of nearest neighbors
    :return: Standard deviation of the residuals
    """
    # Normalize the training and test data    
    X_train = normalize(X_train)
    X_test = normalize(X_test)

    # Find the k nearest neighbors for each test sample
    minimum_indices = find_closests(X_train, X_test, k)
    
    # Predict the target values for the test set 
    pred = regression_Y(minimum_indices, y_train)
    
    # Calculate and retrun the standard deviation of the residuals
    std = calculate_regr_std(pred, y_test)
    
    return std
