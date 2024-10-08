# Function File Group 11
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso

def do_grid_search(X, y, number_of_degrees, model):
    """
    Performs a gridsearch on the given data and returns the gridsearch results
        INPUT:
            X: features of the (training) dataset 
            y: output of the (training) dataset
            number_of_degrees: number of degrees on which to perform the gridsearch
        OUTPUT:
            grid_search: the resulting grid Search
    """

    # Define a grid search for the best alpha hyperparameter 
    param_grid = {'alpha': list(np.arange(0.1, number_of_degrees, 0.21))}

    # Use GridSearchCV to find the best alpha for the Lasso model
    grid_search = GridSearchCV(model, param_grid, cv=10, scoring='neg_mean_squared_error', error_score='raise')

    grid_search.fit(X, y)
    
    return grid_search

def get_grid_search_metrics(grid_search):
    """
    Determines the best lambda and model. Using the best lambda and model calculates the 'results', a.k.a. the performance of said model.

        INPUT:
            grid_search: grid_search 
            X_test: features of the test dataset 
            y: output of the (training) dataset
        OUTPUT:
            results: the resulting grid Search (dictionary of numpy ndarrays)
            best_alpha: the best hyperparameter
    """
    
    # Get the best alpha and model
    best_alpha = grid_search.best_params_['alpha']
    best_model = grid_search.best_estimator_

    print(f'The best alpha parameter is {best_alpha}') 
        
    # Get the results from the grid search
    results = grid_search.cv_results_

    return results, best_alpha

def plot_results_gridsearch(results):
    """
    Plots the negative mean squared error for different lambdas for a Lasso regression model.

        INPUT: 
            results: the resulting grid Search (dictionary of numpy ndarrays)
    """

    # Get the alpha values, mean test scores, and standard deviation of test scores from the grid search results
    alphas = results['param_alpha']
    mean_test_scores = results['mean_test_score']
    std_test_scores = results['std_test_score']
    
    # Plot the test score results (validation accuracy) as a function of alpha with error bars
    plt.figure()
    plt.errorbar(alphas, mean_test_scores, yerr=std_test_scores, fmt='-o', capsize=5)
    plt.xlabel('Alpha')
    plt.ylabel('Negative Mean Squared Error')
    plt.title('Mean Squared Error for different alpha values')
    plt.show()

def lasso_bootstrap(X,y,alphas, n_bootstrap =100):
    """
    Performs Lasso bootstrap on X. It plots the lasso coefficients for different different alpha parameters (with error bars).
        INPUT: 
            X: input features
            y: target variable
            alphas: list of alpha values
            n_bootstrap: number of bootstrap samples
    """

    # Create a matrix to store coefficients 
    # with every row is a bootstrap sample and every column is an alpha-value
    coefficients = np.zeros((n_bootstrap, len(alphas), X.shape[1])) 

    # generate a bootstrap sample from the original dataset, n_bootstrap times
    for b in range(n_bootstrap):
        #Sampling with replacement
        X_bootstrap, y_bootstrap = resample(X,y,random_state=b)

        # loop over each alpha value to fit the lasso model
        for i, alpha in enumerate(alphas):
            # initialize the Lasso regression model with the current alpha value
            lasso = Lasso(alpha=alpha)
            # fit the lasso model to the bootstrap sample
            lasso.fit(X_bootstrap,y_bootstrap)
            # add the coefficients from the fitted model in the coefficients array
            coefficients[b,i,:] = lasso.coef_
            
    # Calculate the mean and standard deviation of the lasso regression coefficients
    # for each feature across all bootstrap samples per alpha value
    mean_coefficients = np.mean(coefficients, axis=0)
    std_coefficients = np.std(coefficients, axis=0)

    # Plot the lasso regression coefficients of each feature for all alpha values
    # with an errorbar(std)
    for feature in range(X.shape[1]):
        plt.errorbar(alphas, 
                     mean_coefficients[:, feature], 
                     yerr=std_coefficients[:, feature], 
                     marker ='.',
                    elinewidth=0.5,
                    linewidth=1,
                    capsize=4)

    plt.xlabel(r'$\alpha$')
    plt.ylabel('Lasso coefficients')
    plt.title('Effect of regularization on Lasso coefficients for each feature') 
    plt.show()
