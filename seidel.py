import numpy as np


def matrix_is_diagonally_dominant(matrix):
    """
    Checking the matrix for diagonal predominance
    """
    diagonally_dominant_matrix = np.array([])
    for i in range(matrix.shape[0]):
        temp = False
        if matrix[i, i] >= np.sum(np.abs(matrix[i, :]) - np.abs(matrix[i, i])):
            temp = True
        diagonally_dominant_matrix = np.append(diagonally_dominant_matrix, temp)
    return diagonally_dominant_matrix.all()


def seidel(matrix, b, LIMIT, eps):
    """
    Gauss–Seidel method: https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method
    """
    if matrix_is_diagonally_dominant(matrix):
        temp = 0
        x = np.zeros_like(b)
        for k in range(1, LIMIT):
            x_new = np.zeros_like(x)
            for i in range(matrix.shape[0]):
                sum_1 = np.matmul(matrix[i, :i], x_new[:i])
                sum_2 = np.matmul(matrix[i, i + 1:], x[i + 1:])
                x_new[i] = (b[i] - sum_1 - sum_2) / matrix[i, i]
            if np.allclose(x, x_new, atol=eps):
                break
            x = x_new
            temp = k
            # print("X_{0} = {1}".format(k, x))
            # print("X_{0}".format(k), end=' | ')
        print('Количество итераций: ', temp)
        print("X = {0}".format(x))
        error = np.dot(matrix, x) - b
        print("Погрешность: {0}".format(error))
        return x, error
    else:
        print("Матрица не обладает свойством диагонального преобладания")
