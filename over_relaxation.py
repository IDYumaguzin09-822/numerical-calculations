import numpy as np


def matrix_is_symmetrical(matrix):
    # upper_matrix = np.triu(matrix)
    main_diag_matrix = np.diag(np.diag(matrix))
    lower_matrix = np.tril(matrix) - main_diag_matrix
    if np.all(matrix == lower_matrix + main_diag_matrix + np.transpose(lower_matrix)):
        print("Матрица симметрична")
        return True
    else:
        print("Матрица не симметрична")
        return False


def matrix_is_positive_definite(matrix):
    if np.all(np.linalg.eigvals(matrix) > 0):
        print("Матрица позитивно определенная")
        return True
    else:
        print("Матрица не позитивно определенная")


def over_relaxation(matrix, b, omega, LIMIT, eps):
    """
    Successive over-relaxation: https://en.wikipedia.org/wiki/Successive_over-relaxation
    :param matrix: input matrix
    :param b:
    :param omega: constant ω > 1, called the relaxation factor
    :param LIMIT:
    :param eps:
    :return:
    """
    if matrix_is_symmetrical(matrix) and matrix_is_positive_definite(matrix):
        x = np.zeros_like(b)
        temp = 0
        for k in range(1, LIMIT):
            x_new = np.zeros_like(x)
            for i in range(matrix.shape[0]):
                sum_1 = np.matmul(matrix[i, :i], x_new[:i])
                sum_2 = np.matmul(matrix[i, i + 1:], x[i + 1:])
                x_new[i] = (b[i] - sum_1 - sum_2) / matrix[i, i] * omega + (1 - omega) * x[i]
            if np.allclose(x, x_new, rtol=eps):
                break
            x = x_new
            temp = k
            # print("X_{0} = {1}".format(k, x))
            # print("X_{0}".format(k), end=' | ')
            # print(k, end='')
        print("Количество итераций: ", temp)
        # print("X = {0}".format(x))
        error = np.dot(matrix, x) - b
        print("Погрешность: {0}".format(error))
    else:
        print("Метод релаксации не сходится")
