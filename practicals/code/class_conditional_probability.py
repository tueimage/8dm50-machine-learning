import numpy as np
import math
    
def normpdf(x_axis, mean, std):
    """ calculates probability density function of standard normal distribution
    outputs the y values of the probability density function
    
    parameters
    ----------
    x_axis : array or float
             x axis of probability density function
    mean : float
           mean of normal distribution
    std : float
          standard deviation of normal distribution
    """
    pdf = []
    if isinstance(x_axis, float): 
        x_axis = np.array([x_axis])
    for x in x_axis:
        exponent = math.exp(-((x-mean)**2 / (2 * std**2 )))
        pdf.append((1 / (math.sqrt(2 * math.pi) * std)) * exponent)
    return pdf