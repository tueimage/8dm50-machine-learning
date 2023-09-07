import numpy as np
la = np.linalg 

def euclidean_squared(u, v):
    """Calculate Euclidean (squared) distance between u and v 
    Euclidean squared: (u-v)^T * (u-v)
    
    Parameters 
    ------------
    u, v : 1D numpy arrays 
        A vector containing floats as values for features of one
        data point 
        
    Returns
    ------------
    distance : float
        Distance as calculated with Euclidean (squared) distance
        between u and v 
    """
    return sum([(x - y)**2 for x, y in zip(u,v)])


def euclidean(u, v):
    """Calculate Euclidean distance between u and v 
    Euclidean distance: sqrt((u-v)^T * (u-v)) = 
                        sqrt(euclidean_squared)
    
    Parameters 
    ------------
    u, v : 1D numpy arrays 
        A vector containing floats as values for features of one
        data point 
        
    Returns
    ------------
    distance : float
        Distance as calculated with Euclidean (squared) distance
        between u and v 
    """
    return np.sqrt(euclidean_squared)


def manhattan(u, v):
    """Calculates Manhattan distance between u and v 
    Manhattan distance: sum(abs(u_p - v_p))
    
    Parameters 
    ------------
    u, v : 1D numpy arrays 
        A vector containing floats as values for features of one
        data point 
        
    Returns
    ------------
    distance : float
        Distance as calculated with Manhattan distance between u and
        v. 
    """
    return sum([abs(x - y) for x, y in zip(u,v)])


def sqrt_einsum(u, v):
    """Calculate distance using np.sqrt(np.einsum()). Fastest way to
    calculate distances according to a comment on Stackoverflow:
    https://stackoverflow.com/a/47775357/8739121.

    Parameters
    ----------
    u, v : arrays
        The ararays between which you want to calculate the
        distances.

    Returns
    -------
    array of shape (distances: flt,)
        Array containing the distances, with:
            distances[0] = distance between u[0] and v[0],
            etc.
    """
    u_min_v = np.array([u]) - np.array([v])
    return np.sqrt(np.einsum("ij,ij->i", u_min_v, u_min_v))

    

#### Some tests
# array_a, array_b = np.random.rand(2,3), np.random.rand(2,3)
# array_a, array_b = np.array([[0, 1], [0, 0]]), np.array([[1, 1], [3, 17]])
# print(np.einsum("ij,ij->i", array_a - array_b, array_a - array_b))
# # print(DistanceCalc().euclidean_squared(array_a[0], array_b[0]))
# print('\n', DistanceCalc.sqrt_einsum(array_a, array_b))