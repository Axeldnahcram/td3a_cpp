"""
Implements of the :epkg:`dot` function in python.
"""
from scipy.stats import linregress

def pylinear_reg(X, y):
    """
    Implements the dot product between two vectors.

    :param va: first vector
    :param vb: second vector
    :return: dot product
    """
    slope, intercept, r_value, p_value, std_err = linregress(X, y)
    return slope
