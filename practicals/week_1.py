import numpy as np
from math import *

def kNearestNeighbour(k, testData, trainingData, trainingLabels):
    """k Nearest neighbour algorithm for classification of data.
    
    :param k: nr of neighbours to be considered.
    :param testData: array of datapoints to be labelled.
    :param trainingData: reference datapoints with known labels.
    :param trainingLabels: array of labels that belong to the trainingData.
    :returns: array of labels for testData.
    """
    newLabels = []
    for newPoint in testData:
        dist = np.linalg.norm(trainingData - newPoint, axis=1)    # Caulculate Euclidean distance per new point 
        nearestNeighbours = np.argsort(dist)[:k]            # Find nearest neighbours per point 
        nearestLabels = np.array(trainingLabels[nearestNeighbours])
        newLabels.append([round(nearestLabels.mean())])
    newLabels = np.array(newLabels)
        
    return newLabels

def normal_pdf(x, mean, standard_deviation): #calculates the probability density function for a normal distribution
    denominator = standard_deviation*sqrt(2*pi)
    exponent = -((x-mean)**2)/(2*standard_deviation**2)
    result = np.exp(exponent)/denominator #numpy.exp to calculate the exponential of all elements in the input array
    
    return result
