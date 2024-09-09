# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:12:26 2024

@author: 20201969
"""
import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer

diabetes = load_diabetes()

breast_cancer = load_breast_cancer()


# the actual implementation is in linear_regression.py,
# here we will just use it to fit a model
from linear_regression import *

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

# load the dataset
# same as before, but now we use all features
X_train = diabetes.data[:300, :]
y_train = diabetes.target[:300, np.newaxis]
X_test = diabetes.data[300:, :]
y_lest = diabetes.target[300:, np.newaxis]

beta = lsq(X_train, y_train)

# print the parameters
print(beta)

"""
k-NN classification
"""

from scipy import stats as st
import matplotlib.pyplot as plt

def normalize(dataset):
    x_min= np.min(dataset)
    x_max= np.max(dataset)
    dataset_norm=(dataset-x_min)/(x_max-x_min)
    return dataset_norm 

def find_closests(X_train, X_test, k):
    #take the sum of the squares of each element per row
    X_train_squared_sum = np.sum(X_train**2, axis = 1) 
    X_test_squared_sum = np.sum(X_test**2, axis = 1)

    #multiply the X_test and x_train matrices
    X_multiplied = np.matmul(X_test, X_train.T) 

    #calculate the distances
    X_test_squared_sum = X_test_squared_sum.reshape(-1,1) 
    dists = np.sqrt(X_test_squared_sum - 2*X_multiplied + X_train_squared_sum)
    # look at this website under'No loops' for the explaination of the formula:
    # https://jaykmody.com/blog/distance-matrices-with-numpy/ 

    #determine the indices of the k nearest neighbors
    #I am not sure if we can use the 'argpartition function'
    minimum_indices = np.argpartition(dists, kth = k-1, axis = -1)[:,:k]
    return minimum_indices

def pred_Y(minimum_indices, y_train):
    # take the data from y_train on the indices defined by minimum_indices
    # and reshape
    nearest_neighbors_outcomes = y_train[[minimum_indices]].reshape(minimum_indices.shape)

    # Take the most frequent predicted outcome 
    outcome = st.mode(np.transpose(nearest_neighbors_outcomes))[0]
    return outcome

def calculate_accuracy(pred, real):
    # reshape to the same shape
    real = real.reshape(pred.shape)

    # outcomes are 0 or 1 so difference between 2 outcomes is 0, 1 or -1
    # calculate the difference squared (to remove sign)
    errors = np.sum((real-pred)**2)

    # devide number of good outocomes devided by total number of outcomes
    accuracy = (real.size-errors)/real.size
    return accuracy
    
def knnClassifier(X_train, X_test, y_train, y_test, k):
    # normalize data
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    
    # find k nearest neighbours
    minimum_indices = find_closests(X_train, X_test, k)

    # predict the outcomes
    pred = pred_Y(minimum_indices, y_train)
    # calculate accuracy
    acc = calculate_accuracy(pred, y_test)
    return acc
    
#Same code as before but then breast cancer data
X_train = breast_cancer.data[:300]
y_train = breast_cancer.target[:300, np.newaxis]
X_test = breast_cancer.data[300:]
y_test = breast_cancer.target[300:, np.newaxis]
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# determine accuracy for different values of k
acc_list = []
for k in range(1,100):
    acc_list.append(knnClassifier(X_train, X_test, y_train, y_test, k))

# plot accuracy of knn classifier for different values of k
xpoints = np.arange(1, 100)
ypoints = np.array(acc_list)

plt.plot(xpoints, ypoints)
plt.title('Accuracy of K-nn classifier for different values of k')
plt.xlabel('Value of k') 
plt.ylabel('Accuracy') 
plt.show()

"""
k-NN regression
"""

def regression_Y(minimum_indices, y_train):
    # take the data from y_train on the indices defined by minimum_indices
    # and reshape
    nearest_neighbors_outcomes = y_train[[minimum_indices]].reshape(minimum_indices.shape)

    # Calculate the mean of the nearest neighbor outcomes
    outcome = np.mean(nearest_neighbors_outcomes, axis=1)
    return outcome
    
def calculate_regr_std(pred, real):
    # reshape to the same shape
    real = real.reshape(pred.shape)
    
    # calculate the difference squared
    errors = sum((real-pred)**2)
    # calculate the standard deviation
    std = (errors/real.size)**0.5
    return std

def knnRegressor(X_train, X_test, y_train, y_test, k):
    # normalize data
    X_train = normalize(X_train)
    X_test = normalize(X_test)

    # find k nearest neighbours
    minimum_indices = find_closests(X_train, X_test, k)
    # predict the outcomes
    pred = regression_Y(minimum_indices, y_train)
    # calculate standard deviation
    std = calculate_regr_std(pred, y_test)
    return std

# prepare the data
X_train = diabetes.data[:300, np.newaxis, 3]
y_train = diabetes.target[:300, np.newaxis]
X_test = diabetes.data[300:, np.newaxis, 3]
y_test = diabetes.target[300:, np.newaxis]
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# determine standard deviation for different values of k
std_list = []
for k in range(1,100):
    std_list.append(knnRegressor(X_train, X_test, y_train, y_test, k))

xpoints = np.arange(1, 100)
ypoints = np.array(std_list)

plt.plot(xpoints, ypoints)
plt.title('Standard deviation of K-nn classifier for different values of k')
plt.xlabel('Value of k') 
plt.ylabel('Standard deviation') 
plt.show()

"""
Class-conditional probability
"""

import seaborn as sns
import matplotlib.pyplot as plt

# Gather data
X = breast_cancer.data
Y = breast_cancer.target[:, np.newaxis]

# Divide Data into malignent and benign
y_bool = np.array(list(map(bool,Y)))
malignant = X[~y_bool]
benign = X[y_bool]

# Create grid for the pltos
fig, axes = plt.subplots(10, 3, figsize=(20, 50))

# Create plots for each feature
for feature in range(X.shape[1]):
    sns.kdeplot(ax=axes[feature//3, feature%3], data=malignant[:, np.newaxis, feature], label="Malignent", fill=True)
    sns.kdeplot(ax=axes[feature//3, feature%3], x=benign[:, feature], label="Benign", fill=True).set(title=f'feature {feature+1}')
    axes[feature//3, feature%3].legend()
