import numpy as np
from scipy.optimize import minimize_scalar
from scipy.linalg import lstsq
from constants import *

__all__ = ['coord_descent', 'gauss_seidel', 'jacobi', 'conjugate_gradient', 'least_squares']


def __set_value_on_position__(array, position, value):
    array[position] = value
    return array


def coord_descent(a_matrix, b_vector, eps):
    def f(var):
        return np.linalg.norm(a_matrix.dot(var) - b_vector)

    def f_scalar(var, position):
        var_copy = np.array(var)
        return lambda t: f(__set_value_on_position__(var_copy, position, t))

    x = np.zeros(len(a_matrix[0]))
    error = float('inf')

    iteration = 1
    while error > eps:
        x_ = np.array(x)
        for i in range(len(x)):
            res = minimize_scalar(f_scalar(x, i), method="Golden")
            x[i] = res.x
        error = np.linalg.norm(x - x_)
        iteration += 1
    return x


def gauss_seidel(a_matrix, b_vector, eps):
    a = np.dot(a_matrix.T, a_matrix)
    b = np.dot(a_matrix.T, b_vector)

    x = np.zeros(a.shape[1])
    l = np.tril(a)
    l_inv = np.linalg.inv(l)
    u = a - l

    error = float('inf')
    while error > eps:
        x_ = x
        x = np.dot(l_inv, b - u.dot(x))
        error = np.linalg.norm(x - x_)

    return x


def jacobi(a_matrix, b_vector, eps):
    a = np.dot(a_matrix.T, a_matrix)
    b = np.dot(a_matrix.T, b_vector)

    x = np.zeros(a.shape[1])
    d = np.diag(np.diag(a))
    d_inv = np.linalg.inv(d)
    r = a - d

    error = float('inf')
    while error > eps:
        x_ = x
        x = np.dot(d_inv, b - r.dot(x))
        error = np.linalg.norm(x - x_)
    return x


def conjugate_gradient(a_matrix, b, eps):
    n = len(a_matrix)
    x = x_last = np.zeros(n, dtype=DEFAULT_FLOAT_TYPE)
    z_last = r_last = b - a_matrix.dot(x_last)
    i = 0
    b_norm = np.linalg.norm(b)
    while True:
        i += 1
        alpha = r_last.dot(r_last) / a_matrix.dot(z_last).dot(z_last)
        x = x_last + alpha * z_last
        r = r_last - alpha * a_matrix.dot(z_last)
        beta = r.dot(r) / r_last.dot(r_last)
        z = r + beta * z_last
        if np.linalg.norm(r) / b_norm < eps:
            break
        else:
            x_last, r_last, z_last = x, r, z
    return x


def least_squares(a_matrix, b, eps):
    return lstsq(a_matrix, b, eps)[0]
