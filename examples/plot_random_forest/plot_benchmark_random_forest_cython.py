"""

.. _l-example-benchmark-random-forest-cython:

Compares random-forest implementations (numpy, cython, c++, sse)
======================================================

:epkg:`sklearn` should be fast:

* :func:`_product <td3a_cpp.random-forest.cpp_py_regular_random-forest.cpp_py_regular_random-forest>`
* :func:`_product <td3a_cpp.random-forest.mp_py_regular_random-forest.mp_py_regular_random-forest>`
* :func:`_product <td3a_cpp.random-forest.nv_py_regular_random-forest.nv_py_regular_random-forest>`

.. contents::
    :local:
"""

import matplotlib.pyplot as plt
import numpy
from pandas import DataFrame, concat
from sklearn.datasets import make_blobs
import pyximport
pyximport.install()


from td3a_cpp.random_forest.method_global import create_forest_prime
from td3a_cpp.random_forest.nv_random_forest import random_forest
from td3a_cpp.tools import measure_time_dim
from sklearn.ensemble import RandomForestClassifier


def get_vectors_cpp(fct, n, h=10, dtype=numpy.float64):
    X, y = make_blobs(n_samples=n, centers=3, n_features=4)
    X = X.astype(numpy.float64)
    print(X.shape)
    y = y.astype(numpy.float64)
    y = y.reshape((n,1))
    data = numpy.concatenate((X,y), axis=1)
    ctxs = [dict(data=data,
                 n_trees=5,
                 random_forest=fct,
                 max_depth=2,
                 max_x=2,
                 x_name=n
                 )
            for n in range(10, n, h)]
    return ctxs

##############################
# numpy matmul
# +++++++++
#

def get_vectors_sklearn(n, h=10, dtype=numpy.float64):
    X, y = make_blobs(n_samples=n, centers=3, n_features=4)
    _, p = X.shape
    X = X.astype(numpy.float64)
    y = y.astype(numpy.float64)
    fct = RandomForestClassifier(n_estimators=5, max_depth=2, max_features=2)
    ctxs = [dict(X=X,
                 y=y,
                 RandomForestClassifier=fct.fit,
                 x_name=n
                 )
            for n in range(10, n, h)]
    return ctxs

ctxs = get_vectors_sklearn(100)
df = DataFrame(list(measure_time_dim('RandomForestClassifier(X,y)', ctxs, verbose=1)))
df['fct'] = 'RandomForestClassifier'
print(df.tail(n=3))
dfs = [df]

##############################
# Several cython matmul
# ++++++++++++++++++
#

for fct in [create_forest_prime, random_forest]:
    ctxs = get_vectors_cpp(fct, 100)

    df = DataFrame(list(measure_time_dim('random_forest(data, n_trees, max_depth, n_trees)', ctxs, verbose=1)))
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
