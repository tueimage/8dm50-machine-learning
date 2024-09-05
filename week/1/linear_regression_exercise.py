import numpy as np
from sklearn.datasets import load_diabetes

# Load the diabetes dataset
diabetes = load_diabetes()

# Split the dataset into training and test sets
X_train = diabetes.data[:300, :]
y_train = diabetes.target[:300, np.newaxis]  # Make sure y is a column vector
X_test = diabetes.data[300:, :]
y_test = diabetes.target[300:, np.newaxis]  # Ensure y_test is a column vector

# Add a column of ones to X_train and X_test to include the intercept term
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Solve for beta using the normal equation
beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Make predictions on the test data
y_pred = X_test @ beta

# Calculate Mean Squared Error (MSE)
mse = np.mean((y_test - y_pred) ** 2)
print(f'Mean Squared Error: {mse}')