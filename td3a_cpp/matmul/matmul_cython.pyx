"""
Many implementations of the matmul product.
See `Cython documentation <http://docs.cython.org/en/latest/>`_.
"""
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy
from libc.stdio cimport printf
from libc.math cimport NAN
from libcpp.vector cimport vector



import numpy
cimport numpy
cimport cython
numpy.import_array()

def matmul_product(va, vb):
    """
    Python matrix product 

    :param va: first matrix
    :param vb: second matrix
    :return: matrix product
    """
    l = []
    for i in range(va.shape[0]):
        l.append(va[i] * vb[i])
    return l


def dmatmul_cython_array(const double[::1] va, const double[::1] vb):
    """
    dot product implemented with matrix of doubles

    :param va: first vector, dtype must be float64
    :param vb: second vector, dtype must be float64
    :return: dot product
    """
    if va.shape[0] != vb.shape[0]:
        raise ValueError("Matrix must have same shape.")
    cdef vector[double] p
    p.reserve(va.shape[0])
    for i in range(va.shape[0]):
        p.push_back(va[i] * vb[i])
    return p

def smatmul_cython_array(const float[::1] va, const float[::1] vb):
    """
    dot product implemented with matrix of floats

    :param va: first vector, dtype must be float32
    :param vb: second vector, dtype must be float32
    :return: dot product
    """
    if va.shape[0] != vb.shape[0]:
        raise ValueError("Matrix must have same shape.")
    cdef vector[double] p
    p.reserve(va.shape[0])
    for i in range(va.shape[0]):
        p.push_back(va[i] * vb[i])
    return p