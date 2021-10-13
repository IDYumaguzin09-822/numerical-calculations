import numpy as np
# from linear_solver import get_matrix, create_random_matrix, test
# from over_relaxation import matrix_is_symmetrical, matrix_is_positive_definite
# matrix, b = get_matrix(10)
# A = create_random_matrix(10)
# x, b = test(A)
# print("Истинное решение: ", x)
# matrix_is_symmetrical(matrix)
# matrix_is_positive_definite(matrix)


def residual_method(matrix, b, eps, LIMIT):
    """
    Read more: https://en.wikipedia.org/wiki/Residual_(numerical_analysis)
    :param matrix: input matrix
    :param b:
    :param eps:
    :param LIMIT:
    :return:
    """
    x = np.zeros_like(b)
    r = np.dot(matrix, x) - b
    k = 0
    while np.max(np.abs(r)) > eps and k < LIMIT:
        tay = np.sum(np.matmul(matrix, r) * r) / np.sum(np.matmul(matrix, r) * np.matmul(matrix, r))
        x = x - tay * r
        r = np.dot(matrix, x) - b
        k = k + 1
    print(f"Iter {k} :x = ", x)
