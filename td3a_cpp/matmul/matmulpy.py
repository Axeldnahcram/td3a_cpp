def pymatmul(matrix_a, matrix_b):
    """
    Implements the dot product between two matrix

    :param matrix_a: first matrix
    :param matrix_b: second matrix
    :return: matrix product
    """
    
    return [(a * b) for a, b in zip(matrix_a, matrix_b)]
