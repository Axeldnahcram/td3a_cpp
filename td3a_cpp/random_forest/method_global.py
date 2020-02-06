# function file
# cpp_regular_linreg - C++ Wrapped Regularized LinearRegression

import os
import subprocess
import numpy as np
import Cython
import pyximport
pyximport.install()


def create_forest_prime(*args, **kwargs):
    from .random_forest import create_forest

    return create_forest(*args, **kwargs)

