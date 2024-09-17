from scipy.stats import norm
import numpy as np

# This function computes the class-conditional probabilities by calculating the mean and standard deviation for a specified feature in the dataset,
# based on the class labels "Benign" and "Malignant". For each class, it extracts the feature values and then returns the corresponding means and standard deviations.

def class_conditional_prob(X, y, feature_idx):
    """
    Compute class-conditional probabilities
    :param X: Training data 
    :param y_test: Test data
    :param feature_idx: Index of features
    :return: Mean and standard deviations of values
    """
    
    classes = np.unique(y)
    means = []
    stds = []
    
    # for each class, compute the mean and standard deviation of the feature
    for c in classes:
        X_c = X[y == c, feature_idx]
        means.append(np.mean(X_c))
        stds.append(np.std(X_c))
        
    return means, stds