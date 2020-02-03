"""

.. _l-example-matmul-cython:

Compares matmul implementations (numpy, cython, c++, sse)
======================================================

:epkg:`numpy` has a very fast implementation of
the matmul product. It is difficult to be better and very easy
to be slower. This example looks into a couple of slower
implementations with cython. The tested functions are
the following:

* :func:`matmul_product <td3a_cpp.tutorial.matmul_cython.matmul_product>`
* :func:`matmul_cython_array <td3a_cpp.tutorial.matmul_cython.matmul_cython_array>`

.. contents::
    :local:
"""

import numpy
import matplotlib.pyplot as plt
from pandas import DataFrame, concat
from td3a_cpp.matmul.matmul_cython import (
    matmul_product, matmul_cython_array,
)
from td3a_cpp.matmul.matmulpy import pymatmul
from td3a_cpp.tools import measure_time_dim


def get_vectors(fct, n, h=100, dtype=numpy.float64):
    ctxs = [dict(va=numpy.random.randn(n).astype(dtype),
                 vb=numpy.random.randn(n).astype(dtype),
                 matmul=fct,
                 x_name=n)
            for n in range(10, n, h)]
    return ctxs

##############################
# numpy matmul
# +++++++++
#


ctxs = get_vectors(numpy.matmul, 10000)
df = DataFrame(list(measure_time_dim('matmul(va, vb)', ctxs, verbose=1)))
df['fct'] = 'numpy.matmul'
print(df.tail(n=3))
dfs = [df]

##############################
# Several cython matmul
# ++++++++++++++++++
#

for fct in [matmul_product, matmul_cython_array]:
    ctxs = get_vectors(fct, 10000 if fct.__name__ != 'matmul_product' else 1000)

    df = DataFrame(list(measure_time_dim('matmul(va, vb)', ctxs, verbose=1)))
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
