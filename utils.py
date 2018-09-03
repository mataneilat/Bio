
"""
    Module containing general utilities
"""
import unittest
import numpy as np

class Singleton(type):
    """
    An implementation of the singleton design pattern
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def upper_triangular_no_diagonal(mat):
    """
    Returns the upper-triangular elements of the given square matrix, excluding the diagonal

    :param mat: The matrix
    :return:    Upper-triangular elements, excluding diagonal
    """
    (n, m) = mat.shape

    if n != m:
        raise ValueError("The matrix must be a square matrix")

    return mat[np.triu_indices(m, 1)]


def partition_generator(l, n):
    """
    Generates a partition of the given list to chunks of size n

    :param l:   The list to partition
    :param n:   The chunk size

    :return:    List of lists representing the chunks
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_maximum_sub_matrix_around_diagonal_element(matrix, i, size):
    """
    Returns the maximum square sub-matrix around the diagonal element (i,i) which is at most of dimensions (size,size)

    :param matrix: The matrix
    :param i:   The diagonal element index
    :param size:    The maximum dimensions of the sub-matrix

    :return:    The maximum square sub-matrix around the diagonal element (i,i) which is at most of dimensions (size,size)
    """
    (n, m) = matrix.shape
    if n != m:
        return None
    sub_matrix_size = min(min(i ,size), min(m-i-1, size))

    return matrix[i - sub_matrix_size:i + sub_matrix_size + 1, i - sub_matrix_size:i + sub_matrix_size + 1]


class TestUtilities(unittest.TestCase):
    """
    Test methods
    """
    def test_partition_generator(self):
        self.assertEqual(list(partition_generator(list(range(5, 18)), 5)), [[5,6,7,8,9],[10,11,12,13,14],[15,16,17]])


if __name__ == '__main__':
    unittest.main()
