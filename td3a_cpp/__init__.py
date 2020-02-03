# coding: utf-8
"""
Template to develop a python module using
:epkg:`cython` and :epkg:`openmp`.
"""

__version__ = "0.1.1"
__author__ = "Xavier Dupré"


def check_tutorial(verbose=1):
    import pprint
    import numpy
    from .tutorial import pydot, cblas_ddot, cblas_sdot
    from .tutorial.dot_cython import (
        dot_product, ddot_cython_array,
        ddot_cython_array_optim, ddot_array,
        ddot_array_16, ddot_array_16_sse
    )
    from .tutorial.dot_cython import (
        sdot_cython_array,
        sdot_cython_array_optim, sdot_array,
        sdot_array_16, sdot_array_16_sse
    )
    from .tools import measure_time
    rows = []

    # double
    if verbose > 0:
        print("\ndouble\n")

    va = numpy.random.randn(100).astype(numpy.float64)
    vb = numpy.random.randn(100).astype(numpy.float64)
    fcts = [
        ('pydot', pydot, 1),
        ('numpy.dot', numpy.dot),
        ('ddot', cblas_ddot),
        ('dot_product', dot_product),
        ('ddot_cython_array', ddot_cython_array),
        ('ddot_cython_array_optim', ddot_cython_array_optim),
        ('ddot_array', ddot_array),
        ('ddot_array_16', ddot_array_16),
        ('ddot_array_16_sse', ddot_array_16_sse),
    ]

    for tu in fcts:
        name, fct = tu[:2]
        ctx = {'va': va, 'vb': vb, 'fctdot': fct}
        if len(tu) == 3:
            res = measure_time('fctdot(va, vb)', ctx, repeat=tu[2])
        else:
            res = measure_time('fctdot(va, vb)', ctx)
        res['name'] = name
        if verbose > 0:
            pprint.pprint(res)
        rows.append(res)

    # float
    if verbose > 0:
        print("\nfloat\n")

    va = numpy.random.randn(100).astype(numpy.float32)
    vb = numpy.random.randn(100).astype(numpy.float32)
    fcts = [
        ('pydot', pydot, 1),
        ('numpy.dot', numpy.dot),
        ('sdot', cblas_sdot),
        ('sdot_cython_array', sdot_cython_array),
        ('sdot_cython_array_optim', sdot_cython_array_optim),
        ('sdot_array', sdot_array),
        ('sdot_array_16', sdot_array_16),
        ('sdot_array_16_sse', sdot_array_16_sse),
    ]

    for tu in fcts:
        name, fct = tu[:2]
        ctx = {'va': va, 'vb': vb, 'fctdot': fct}
        if len(tu) == 3:
            res = measure_time('fctdot(va, vb)', ctx, repeat=tu[2])
        else:
            res = measure_time('fctdot(va, vb)', ctx)
        res['name'] = name
        if verbose > 0:
            pprint.pprint(res)
        rows.append(res)

    return rows

def check_matmul(verbose=1):
    import pprint
    import numpy
    from .matmul import pymatmul
    from .matmul.matmul_cython import (
        matmul_product, dmatmul_cython_array
    )
    from .matmul.matmul_cython import (
        smatmul_cython_array
    )
    from .tools import measure_time
    rows = []

    # double
    if verbose > 0:
        print("\ndouble\n")

    va = numpy.random.randn(100).astype(numpy.float64)
    vb = numpy.random.randn(100).astype(numpy.float64)
    fcts = [
        ('pymatmul', pymatmul, 1),
        ('numpy.matmul', numpy.matmul),
        ('matmul_product', matmul_product),
        ('dmatmul_cython_array', dmatmul_cython_array)
    ]

    for tu in fcts:
        name, fct = tu[:2]
        ctx = {'va': va, 'vb': vb, 'fctmatmul': fct}
        if len(tu) == 3:
            res = measure_time('fctmatmul(va, vb)', ctx, repeat=tu[2])
        else:
            res = measure_time('fctmatmul(va, vb)', ctx)
        res['name'] = name
        if verbose > 0:
            pprint.pprint(res)
        rows.append(res)

    # float
    if verbose > 0:
        print("\nfloat\n")

    va = numpy.random.randn(100).astype(numpy.float32)
    vb = numpy.random.randn(100).astype(numpy.float32)
    fcts = [
        ('pymatmul', pymatmul, 1),
        ('numpy.matmul', numpy.matmul),
        ('smatmul_cython_array', smatmul_cython_array)
    ]

    for tu in fcts:
        name, fct = tu[:2]
        ctx = {'va': va, 'vb': vb, 'fctmatmul': fct}
        if len(tu) == 3:
            res = measure_time('fctmatmul(va, vb)', ctx, repeat=tu[2])
        else:
            res = measure_time('fctmatmul(va, vb)', ctx)
        res['name'] = name
        if verbose > 0:
            pprint.pprint(res)
        rows.append(res)

    return rows

def check_linreg(verbose=1):
    import pprint
    import numpy as np
    from .linear_regression import pylinear_reg
    from .linear_regression.linear_reg_cython import (
        py_regularized_linreg
    )
    from .tools import measure_time
    rows = []

    # double
    if verbose > 0:
        print("\ndouble\n")
    X = np.random.random(10)
    y = np.random.random(10)
    fcts = [
        ('pylinear_reg', pylinear_reg, 1),
        ('py_regularized_linreg', py_regularized_linreg)
    ]
    for tu in fcts:
        name, fct = tu[:2]
        ctx = {'X': X, 'y': y, 'fctlinearreg': fct}
        if len(tu) == 3:
            res = measure_time('fctlinearreg(X, y)', ctx, repeat=tu[2])
        else:
            res = measure_time('fctlinearreg(X, y)', ctx)
        res['name'] = name
        if verbose > 0:
            pprint.pprint(res)
        rows.append(res)
        
    return rows

def check(verbose=1):
    """
    Runs a couple of functions to check the module is working.

    :param verbose: 0 to hide the standout output
    :return: list of dictionaries, result of each test
    """
    rows = []
    rows.append(check_matmul(verbose))
    rows.append(check_tutorial(verbose))
    rows.append(check_linreg(verbose))
    return rows
