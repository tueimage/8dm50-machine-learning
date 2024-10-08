# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:58:14 2024

@author: 20203129
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn import neighbors


def polynomial_regression_analysis(X,y,number_of_degrees, plot_name):
    """
    Perform polynomial regression analysis and 
    visualize the validation R2 score as a function of the polynomial order
    X: input data
    y: target values
    number_of_degrees: number of degrees for the polynomial to test
    plot_name: name to add to plot
    """
    # split the data and targets into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=40)

    # define a pipeline with 3 steps; scaling, polynomial features, and linear regression
    model = Pipeline([
                      ("scaler", StandardScaler()),
                      ("poly", PolynomialFeatures()),
                      ("regression", LinearRegression())
                     ])
    # define a grid search for the best polynomial degree 
    param_grid = {
        'poly__degree': list(range(1,number_of_degrees)) 
    }

    # use GridSearchCV to find the best degree for the polynomial
    grid_search = GridSearchCV(model, param_grid, cv=10, scoring='r2', error_score='raise')

    # train GridSearchCV  on the training dataset
    grid_search.fit(X_train, y_train)

    # get the best polynomial degree and model
    best_degree = grid_search.best_params_['poly__degree']
    best_model = grid_search.best_estimator_

    # make predictions using the best model on the test data
    y_pred = best_model.predict(X_test)

    # evaluate the model's performance on an independent test set using R2 score
    r2 = r2_score(y_test, y_pred)

    print(f'the best degree of the polynomial on the {plot_name} is', best_degree, 
          'and the R2 score on a test set is',r2)

    # get the results from the grid search
    results = grid_search.cv_results_

    # get the polynomial degrees and test scores from the grid search results
    degrees = results['param_poly__degree']
    mean_test_scores = results['mean_test_score']  

    # plot the test score results (validation accuracy) as a function of the degree (polynomial order)
    plt.figure()
    plt.plot(degrees, mean_test_scores)
    plt.xlabel('polynomial order')
    plt.ylabel('validation ' r'$R^2$' ' score')
    plt.title (f'{plot_name} \n validation accuracy vs polynomial order')
    plt.axvline(x=best_degree, color='purple',ls='--', label='best polynomial degree')
    plt.legend()
    plt.show()

def knn_classfier(k, X, y):
    """
    Perform k-Nearest Neighbors classification
    k: number of nearest neighbors
    X: input data
    y: target values
    """
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    scaler = StandardScaler()
    model_knn = Pipeline([
                 ("scaler", scaler),
                 ("knn", knn)
                ])  
    
    # train the model using the training dataset
    model_knn.fit(X, y)  
    return model_knn
