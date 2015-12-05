from itertools import accumulate, chain
from operator import add, mul

import matplotlib.pyplot as plt
import numpy as np
from scipy import special

from functional_restoration.private.constants import DEFAULT_METHOD, DEFAULT_FLOAT_TYPE, CHEBYSHEV, LEGENDRE, LAGUERRE, \
    HERMITE
from functional_restoration.private.minimize import *

__all__ = ['minimize_equation', 'normalize_x_matrix', 'normalize_y_matrix',
           'make_b_matrix', 'get_polynom_function', 'make_lambdas', 'make_split_lambdas',
           'x_i_j_dimensions', 'make_psi', 'make_a_small_matrix', 'make_f_i', 'make_c_small',
           'make_f', 'make_real_f', 'show_plots', 'convert_degrees_to_string']


def minimize_equation(a_matrix, b_vector, eps, method=DEFAULT_METHOD):
    if method is 'cdesc':
        return coord_descent(a_matrix, b_vector, eps)
    elif method is 'seidel':
        return gauss_seidel(a_matrix, b_vector, eps)
    elif method is 'jacobi':
        return jacobi(a_matrix, b_vector, eps)
    elif method is 'conj':
        return conjugate_gradient(a_matrix.T.dot(a_matrix), a_matrix.T.dot(b_vector), eps)
    else:
        return least_squares(a_matrix, b_vector, eps)


def normalize_vector(v):
    v_min, v_max = np.min(v), np.max(v)
    l = v_max - v_min
    scales = np.array([v_min, l])
    normed_v = (v - v_min) / l
    return normed_v, scales


def normalize_x_matrix(x_matrix):
    x_normed, x_scales = [], []
    for x_i in x_matrix:
        x_i_normed = []
        scales_x_i = []
        for x_i_j in x_i:
            current_normed_vector, current_scales = normalize_vector(x_i_j)
            x_i_normed.append(current_normed_vector)
            scales_x_i.append(current_scales)
        x_normed.append(np.vstack(x_i_normed))
        x_scales.append(np.vstack(scales_x_i))
    return np.array(x_normed), np.array(x_scales)


def normalize_y_matrix(y_matrix):
    y_normed, y_scales = [], []
    for y_i in y_matrix:
        current_normed_vector, current_scales = normalize_vector(y_i)
        y_normed.append(current_normed_vector)
        y_scales.append(current_scales)
    return np.vstack(y_normed), np.vstack(y_scales).astype(DEFAULT_FLOAT_TYPE)


def make_b_matrix(y_matrix, weights):
    dim_y = len(y_matrix)
    if weights is 'average':
        return np.tile((np.max(y_matrix, axis=0) + np.min(y_matrix, axis=0)) / 2, (dim_y, 1))
    else:
        return np.array(y_matrix)


def get_polynom_function(poly_type):
    if poly_type is CHEBYSHEV:
        return special.eval_chebyt
    elif poly_type is LEGENDRE:
        return special.eval_legendre
    elif poly_type is LAGUERRE:
        return special.eval_laguerre
    elif poly_type is HERMITE:
        return special.eval_hermite
    else:
        return special.eval_chebyt


def make_lambdas(a_matrix, b_matrix, eps, method):
    return np.array([minimize_equation(a_matrix, b, eps, method) for b in b_matrix])


def make_split_lambdas(a_matrix, b_matrix, eps, method, dims_x_i, p):
    lambdas = []

    for i in range(len(b_matrix)):
        start = 0
        lambdas_i = []
        for end in accumulate(dims_x_i * p):
            lambdas_i.append(minimize_equation(a_matrix[:, start:end], b_matrix[i], eps, method))
            start = end
        lambdas.append(np.hstack(lambdas_i))

    return np.array(lambdas)


def x_i_j_dimensions(x_matrix):
    return [i for i in range(len(x_matrix)) for j in x_matrix[i]]


def __calculate_psi_line__(line, maper_x_i_j, p):
    psi_list = []
    cursor = 0
    for i in maper_x_i_j:
        psi_list.append(np.sum(line[cursor:cursor + p[i]]))
        cursor += p[i]
    return psi_list


def make_psi(a_matrix, x_matrix, lambdas, p):
    psi = []
    n = len(x_matrix[0][0])
    maper_x_i_j = [i for i in range(len(x_matrix)) for j in x_matrix[i]]

    for i in range(len(lambdas)):
        psi_i = []
        for q in range(n):
            psi_i.append(__calculate_psi_line__(a_matrix[q] * lambdas[i], maper_x_i_j, p))
        psi.append(psi_i)
    return np.array(psi)


def make_a_small_matrix(y_matrix, psi_matrix, eps, method, dims_x_i):
    a = []
    for i in range(len(y_matrix)):
        a_i_k = []
        last = 0
        for j in accumulate(dims_x_i, func=add):
            a_i_k.append(minimize_equation(psi_matrix[i][:, last:j], y_matrix[i], eps, method))
            last = j
        a.append(list(chain(a_i_k)))
    return np.array(a)


def make_f_i(a_small, psi_matrix, dims_x_i):
    f_i = []
    for i in range(len(a_small)):
        f_i_i = []
        last = 0
        for count, j in zip(range(len(dims_x_i)), accumulate(dims_x_i, func=add)):
            f_i_i.append(np.sum(a_small[i][count] * psi_matrix[i][:, last:j], axis=1))
            last = j
        f_i.append(f_i_i)
    return np.array(f_i)


def make_c_small(y_matrix, f_i, eps, method):
    return np.array([minimize_equation(np.column_stack(i), j, eps, method) for i, j in zip(f_i, y_matrix)])


def make_f(f_i, c):
    return np.array([np.sum(list(map(mul, i, j)), axis=0) for i, j in zip(f_i, c)])


def make_real_f(y_scales, f):
    real_f = []
    for i, j in zip(y_scales, f):
        shift, zoom = np.min(i), np.max(i)
        real_f.append(j * zoom + shift)
    return np.array(real_f)


def __plot_of_y_y_approximation__(y, y_approximation, name='no_name'):
    plt.title(name)
    plt.plot(y, 'b')
    plt.plot(y_approximation, 'r')
    plt.show()


def show_plots(y, y_approximation):
    for i, j, k in zip(y, y_approximation, range(len(y))):
        __plot_of_y_y_approximation__(i, j, 'Y-{:d}'.format(k + 1))


def convert_degrees_to_string(degrees):
    return ' '.join('X{:d} - {:d}'.format(i + 1, degrees[i]) for i in range(len(degrees)))
