import numpy as np

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
