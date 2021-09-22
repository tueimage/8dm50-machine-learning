from math import *
import numpy as np

def normal_pdf(x, mean, standard_deviation): #calculates the probability density function for a normal distribution
    denominator = standard_deviation*sqrt(2*pi)
    exponent = -((x-mean)**2)/(2*standard_deviation**2)
    result = np.exp(exponent)/denominator #numpy.exp to calculate the exponential of all elements in the input array
    
    return result