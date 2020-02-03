cdef extern from "linear_reg_cython.hpp":
    double* coord_desc(double *x, double *y, int num_samples, int num_features, double *beta,
                       double alpha, double L1_ratio, int max_iter, double tol)