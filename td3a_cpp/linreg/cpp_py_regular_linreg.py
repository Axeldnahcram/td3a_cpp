# function file
# cpp_regular_linreg - C++ Wrapped Regular LinearRegression

import os
import subprocess
import numpy as np
import Cython
import pyximport
pyximport.install()


def cpp_regular_linreg(X, y, beta_0, alpha, L1_ratio, max_iter, tol, *args, **kwargs):
    """
    C++ regular linear regression 

    :return: slope of the regression
    """

    from .cy_regularized_linreg import py_regularized_linreg

    beta = beta_0
    num_samples, num_features = X.shape
    X_resh = X.reshape((np.prod(X.shape), ))
    py_regularized_linreg(X_resh,
                          y,
                          num_samples,
                          num_features,
                          beta,
                          alpha,
                          L1_ratio,
                          max_iter,
                          tol)

    return beta
