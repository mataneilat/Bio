import unittest
import numpy as np

class Range:

    def __init__(self, start_idx, end_idx):
        self.start_idx = start_idx
        self.end_idx = end_idx

    def __repr__(self):
        return "<%s:%s>" % (self.start_idx, self.end_idx)

    def is_within(self, other_range):
        return self.start_idx >= other_range.start_idx and self.end_idx <= other_range.end_idx


def upper_triangular_no_diagonal(mat):
    (n, m) = mat.shape

    if n != m:
        raise ValueError("The matrix must be a square matrix")

    return mat[np.triu_indices(m, 1)]

def normalize_matrix(M):
    xmax, xmin = M.max(), M.min()
    return (M - xmin)/(xmax - xmin)


def partition_generator(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_block_metrics_around_center(matrix):
    (r1,r2) = matrix.shape

    if r1 != r2:
        raise ValueError("The matrix must be a square matrix")
    if r1 % 2 != 1:
        raise ValueError("The matrix dimension must be odd")

    pivot = int((r1 - 1) / 2)

    left_up = matrix[0:pivot,0:pivot].copy()
    right_bottom = matrix[pivot+1:r1,pivot+1:r1].copy()
    cross = matrix[0:pivot,pivot+1:r1].copy()

    return left_up, right_bottom, cross

def get_maximum_sub_matrix_around_diagonal_element(matrix, i, size):
    (n, m) = matrix.shape
    if n != m:
        return None
    sub_matrix_size = min(min(i ,size), min(m-i-1, size))

    return matrix[i - sub_matrix_size:i + sub_matrix_size + 1, i - sub_matrix_size:i + sub_matrix_size + 1]


def fold_matrix_around_diagonal_element(matrix, i, max_delta=5):
    sub_matrix = get_maximum_sub_matrix_around_diagonal_element(matrix, i, max_delta)
    return fold_matrix_around_center(sub_matrix, max_delta)

def flattem_matrix_around_diagonal_element(matrix, i, max_delta=5):
    sub_matrix = get_maximum_sub_matrix_around_diagonal_element(matrix, i, max_delta)
    return sub_matrix.flatten()

def fold_matrix_around_center(matrix, max_delta=5):
    (n, m) = matrix.shape
    if n != m or m % 2 != 1:
        return None
    pivot = int((m - 1) / 2)
    folded = [matrix[pivot,pivot]]
    gen = (j for j in range(max_delta) if pivot - j >= 0)
    for j in gen:
        l = pivot - j
        for k in range(pivot - j, pivot + j):
            folded.append(matrix[k,l])
        k = pivot + j
        for l in range(pivot - j, pivot + j):
            folded.append(matrix[k,l])
        l = pivot + j
        for k in range(pivot + j, pivot - j, -1):
            folded.append(matrix[k,l])
        k = pivot - j
        for l in range(pivot + j, pivot - j, -1):
            folded.append(matrix[k,l])
    return folded


def flatten_list(l):
    return [item for sublist in l for item in sublist]

class TestUtilities(unittest.TestCase):

    def test_fold(self):
        matrix = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]])
        self.assertEqual(fold_matrix_around_diagonal_element(matrix, 0), [1])
        self.assertEqual(fold_matrix_around_diagonal_element(matrix, 1), [6, 1, 5, 9 , 10, 11, 7, 3, 2])
        self.assertEqual(fold_matrix_around_diagonal_element(matrix, 2), [11, 6, 10, 14, 15, 16, 12, 8, 7])
        self.assertEqual(fold_matrix_around_diagonal_element(matrix, 3), [16])

    def test_partition_generator(self):
        self.assertEqual(list(partition_generator(list(range(5, 18)), 5)), [[5,6,7,8,9],[10,11,12,13,14],[15,16,17]])


if __name__ == '__main__':
    unittest.main()
