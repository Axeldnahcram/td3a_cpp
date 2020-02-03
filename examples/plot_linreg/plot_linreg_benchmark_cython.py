"""

.. _l-example-benchmark-linreg-cython:

Compares linreg implementations (numpy, cython, c++, sse)
======================================================

:epkg:`sklearn` should be fast:

* :func:`_product <td3a_cpp.linreg.cpp_py_regular_linreg.cpp_py_regular_linreg>`
* :func:`_product <td3a_cpp.linreg.mp_py_regular_linreg.mp_py_regular_linreg>`
* :func:`_product <td3a_cpp.linreg.nv_py_regular_linreg.nv_py_regular_linreg>`

.. contents::
    :local:
"""

import matplotlib.pyplot as plt
import numpy
from pandas import DataFrame, concat
from sklearn.datasets import make_blobs

from td3a_cpp.linreg.cpp_py_regular_linreg import cpp_py_regular_linreg
from td3a_cpp.linreg.mp_py_regular_linreg import mp_py_regular_linreg
from td3a_cpp.linreg.nv_py_regular_linreg import nv_py_regular_linreg
from td3a_cpp.tools import measure_time_dim
from sklearn.linear_model import ElasticNet

alpha = 1
L1_ratio = 0.5
max_iter = 100
tol = 1e-5

def get_vectors(fct, n, h=100, dtype=numpy.float64):
    X, y = make_blobs(n_samples=n, centers=3, n_features=2)
    _, p = X.shape
    X = X.astype(numpy.float64)
    y = y.astype(numpy.float64)

    ctxs = [dict(X=X,
                 y=y,
                 linear_regression=fct,
                 beta=numpy.zeros(p),
                 alpha=alpha,
                 L1_ratio=L1_ratio,
                 max_iter=max_iter,
                 tol=tol,
                 num_samples=n,
                 num_features=p,
                 x_name=n
                 )
            for n in range(10, n, h)]
    return ctxs

##############################
# numpy matmul
# +++++++++
#

def get_vectors_elastic(n, h=100, dtype=numpy.float64):
    X, y = make_blobs(n_samples=n, centers=3, n_features=2)
    _, p = X.shape
    X = X.astype(numpy.float64)
    y = y.astype(numpy.float64)
    fct = ElasticNet(alpha=alpha, l1_ratio=L1_ratio)
    ctxs = [dict(X=X,
                 y=y,
                 linear_regression=fct.fit,
                 x_name=n
                 )
            for n in range(10, n, h)]
    return ctxs

ctxs = get_vectors_elastic(1000)
df = DataFrame(list(measure_time_dim('linear_regression(X, y)', ctxs, verbose=1)))
df['fct'] = 'LinearRegression'
print(df.tail(n=3))
dfs = [df]

##############################
# Several cython matmul
# ++++++++++++++++++
#

for fct in [nv_py_regular_linreg, cpp_py_regular_linreg]:
    ctxs = get_vectors(fct, 1000)

    df = DataFrame(list(measure_time_dim('linear_regression(X, y, beta, alpha, L1_ratio, max_iter, tol, num_samples, num_features)', ctxs, verbose=1)))
    df['fct'] = fct.__name__
    dfs.append(df)
    print(df.tail(n=3))

#############################
# Let's display the results
# +++++++++++++++++++++++++

cc = concat(dfs)
cc['N'] = cc['x_name']

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
cc.pivot('N', 'fct', 'average').plot(
         logy=True, ax=ax[0])
cc.pivot('N', 'fct', 'average').plot(
         logy=True, logx=True, ax=ax[1])
ax[0].set_title("Comparison of cython sdot implementations")
ax[1].set_title("Comparison of cython sdot implementations")



plt.show()
