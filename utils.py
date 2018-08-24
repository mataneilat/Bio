import unittest
import numpy as np


def normalize_matrix(M):
    xmax, xmin = M.max(), M.min()
    return (M - xmin)/(xmax - xmin)

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


class TestFoldMethods(unittest.TestCase):

    def test_fold(self):
        matrix = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]])
        self.assertEqual(fold_matrix_around_diagonal_element(matrix, 0), [1])
        self.assertEqual(fold_matrix_around_diagonal_element(matrix, 1), [6, 1, 5, 9 , 10, 11, 7, 3, 2])
        self.assertEqual(fold_matrix_around_diagonal_element(matrix, 2), [11, 6, 10, 14, 15, 16, 12, 8, 7])
        self.assertEqual(fold_matrix_around_diagonal_element(matrix, 3), [16])

if __name__ == '__main__':
    unittest.main()
