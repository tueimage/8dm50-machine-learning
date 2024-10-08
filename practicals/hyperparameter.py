import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

def Lasso_analysis(X, y, number_of_degrees):
    # Split the data and targets into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # Define a grid search for the best alpha hyperparameter 
    param_grid = {'alpha': list(np.arange(0.1, number_of_degrees, 0.21))}

    model = Lasso(max_iter=1500)

    # Use GridSearchCV to find the best alpha for the Lasso model
    grid_search = GridSearchCV(model, param_grid, cv=10, scoring='neg_mean_squared_error', error_score='raise')

    grid_search.fit(X_train, y_train)

    # Get the best alpha and model
    best_alpha = grid_search.best_params_['alpha']
    best_model = grid_search.best_estimator_

    # Make predictions using the best model on the test data
    y_pred = best_model.predict(X_test)

    # Evaluate the model's performance on an independent test set using MSE
    mse = mean_squared_error(y_test, y_pred)

    print(f'The best alpha parameter is {best_alpha} and the mean squared error score on a test set is {mse}')

    # Get the results from the grid search
    results = grid_search.cv_results_

    # Get the alpha values, mean test scores, and standard deviation of test scores from the grid search results
    alphas = results['param_alpha']
    mean_test_scores = results['mean_test_score']
    std_test_scores = results['std_test_score']

    # Plot the test score results (validation accuracy) as a function of alpha with error bars
    plt.figure()
    plt.errorbar(alphas, mean_test_scores, yerr=std_test_scores, fmt='-o')
    plt.xlabel('Alpha')
    plt.ylabel('Negative Mean Squared Error')
    plt.title('Mean Squared Error for different alpha values')
    plt.show()

# Load the data
gene_expression = pd.read_csv("RNA_expression_curated.csv", sep=',', header=0, index_col=0)
drug_response = pd.read_csv("drug_response_curated.csv", sep=',', header=0, index_col=0)

# Prepare the input and output data
X = gene_expression.values
y = drug_response['YM155'].values

# Experiment with the dataset
Lasso_analysis(X, y, 6)