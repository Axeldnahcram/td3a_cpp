# funciton file
# cpp_py_regular_linreg.py
import os, subprocess
import numpy as np
import Cython
import pyximport; 
pyximport.install()

def cpp_py_regular_linreg(X, y, beta_0, alpha, L1_ratio, max_iter, tol, *args, **kwargs):
    
    #subprocess.Popen(['python', 'setup.py', 'clean'])
    #subprocess.Popen(['python', 'setup.py', '-q', 'build_ext', '--inplace'])

    
    from .cy_regularized_linreg import py_regularized_linreg
     
    beta = beta_0
    num_samples, num_features = X.shape
    X_resh = X.reshape((np.prod(X.shape), ))
    py_regularized_linreg(X_resh, y,
                            num_samples, num_features,
                            beta,
                            alpha, L1_ratio,
                            max_iter, tol)
    
    return beta