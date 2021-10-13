import numpy as np
from seidel import seidel
from over_relaxation import over_relaxation


def u_func(x):
    return (2 - x) * (x + 2)


def p_func(x, gamma=3):
    return 1 + x ** gamma


def g_func(x):
    return x + 1


def f_func(x, gamma=3):
    return -2 * (1 + (1 + gamma) * x**gamma) + g_func(x) * u_func(x)


def a_i_func(i, h):
    return (p_func(i * h - h) + p_func(i * h)) / 2


def get_matrix(n, mu1=0, mu4=5):
    h = 1 / n
    main_diag, down_diag, upper_diag, b = np.array([]), np.array([]), np.array([]), np.array([])
    for i in range(1, n):
        main_diag = np.append(main_diag, a_i_func(i, h) + a_i_func(i + 1, h) + h**2 * g_func(i))
        down_diag = np.append(down_diag, -a_i_func(i, h))
        upper_diag = np.append(upper_diag, -a_i_func(i + 1, h))
        b = np.append(b, f_func(i * h) * h**2)

    main_diag = np.insert(main_diag, 0, a_i_func(1, h) + 1 / 2 * h**2 * g_func(0))
    main_diag = np.insert(main_diag, main_diag.size,
                          a_i_func(n, h) + 1 / 2 * h**2 * g_func(n) - h * p_func(n * h))

    upper_diag = np.insert(upper_diag, 0, -a_i_func(1, h))
    down_diag = np.insert(down_diag, down_diag.size, -a_i_func(n, h))

    b = np.insert(b, 0, 1 / 2 * h**2 * f_func(0) - h * mu1 * p_func(0))
    b = np.insert(b, b.size, 1 / 2 * h**2 * f_func(n) - h * mu4 * p_func(n))

    A = np.zeros((n + 1, n + 1)) + np.diag(main_diag) + np.diag(upper_diag, 1) + np.diag(down_diag, -1)
    return A, b


def create_random_matrix(n):
    """
    Creates a random matrix with a predominance of the diagonal,
    symmetric and positive definite
    :param n: the dimension of the matrix
    :return:
    """
    A = np.random.random((n, n))
    d = np.array([])
    for i in range(A.shape[0]):
        d = np.append(d, np.sum(A[i, :]))
    A = A + np.diag(d)
    return A * np.transpose(A)


def test(matrix):
    x = np.random.randint(-10, 10, matrix.shape[0])
    b = np.matmul(matrix, x)
    print(x)
    print(b)
    # print("System of equations:")
    # for i in range(matrix.shape[0]):
    #     row = ["{0:3g} * {1}".format(matrix[i, j], x[j]) for j in range(matrix.shape[1])]
    #     print("[{0}] = [{1:3g}]".format(" + ".join(row), b[i]))
    return x, b


def main():
    gamma = 3
    n = 10
    h = 1 / n
    eps = h ** 3
    mu1 = 0
    mu4 = 5

    A, b = get_matrix(n, mu1, mu4)
    # A = create_random_matrix(50)
    x, b = test(A)
    seidel(A, b, 1000, eps)
    print("-" * 30)

    omega = np.linspace(1, 2, 11)
    for t in omega:
        print("-" * 30)
        print(t)
        over_relaxation(A, b, t, 1000, eps)

    from residual_method import residual_method
    residual_method(A, b, eps, 1000)


if __name__ == '__main__':
    main()
    # test(create_random_matrix(10))