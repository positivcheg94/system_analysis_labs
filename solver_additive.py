from copy import deepcopy
from itertools import accumulate, chain, product
from operator import add, mul

import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy import special

from minimize import *
from constants import DEFAULT_FLOAT_TYPE, DEFAULT_METHOD, CHEBYSHEV, LAGUERRE, LEGENDRE, HERMITE, CONST_EPS
from polynom_representation import polynom_representation_add


def __convert_degrees_to_string__(degrees):
    return ' '.join('X{:d} - {:d}'.format(i + 1, degrees[i]) for i in range(len(degrees)))


def __minimize_equation__(a_matrix, b_vector, eps, method=DEFAULT_METHOD):
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


def __normalize_vector__(v):
    v_min, v_max = np.min(v), np.max(v)
    l = v_max - v_min
    scales = np.array([v_min, l])
    normed_v = (v - v_min) / l
    return normed_v, scales


def __normalize_x_matrix__(x_matrix):
    x_normed, x_scales = [], []
    for x_i in x_matrix:
        x_i_normed = []
        scales_x_i = []
        for x_i_j in x_i:
            current_normed_vector, current_scales = __normalize_vector__(x_i_j)
            x_i_normed.append(current_normed_vector)
            scales_x_i.append(current_scales)
        x_normed.append(np.vstack(x_i_normed))
        x_scales.append(np.vstack(scales_x_i))
    return np.array(x_normed), np.array(x_scales)


def __normalize_y_matrix__(y_matrix):
    y_normed, y_scales = [], []
    for y_i in y_matrix:
        current_normed_vector, current_scales = __normalize_vector__(y_i)
        y_normed.append(current_normed_vector)
        y_scales.append(current_scales)
    return np.vstack(y_normed), np.vstack(y_scales).astype(np.float64)


def __make_b_matrix__(y_matrix, weights):
    dim_y = len(y_matrix)
    if weights is 'average':
        return np.tile((np.max(y_matrix, axis=0) + np.min(y_matrix, axis=0)) / 2, (dim_y, 1))
    else:
        return np.array(y_matrix)


def __get_polynom_function__(poly_type):
    if poly_type is CHEBYSHEV:
        return special.eval_sh_chebyt
    elif poly_type is LEGENDRE:
        return special.eval_sh_legendre
    elif poly_type is LAGUERRE:
        return special.eval_laguerre
    elif poly_type is HERMITE:
        return special.eval_hermite
    else:
        return special.eval_sh_chebyt


def __make_a_matrix__(x, p, polynom):
    n = len(x[0][0])
    a_matrix = np.array(
        [[polynom(p_j, j[k]) for x_i in range(len(x)) for j in x[x_i] for p_j in range(p[x_i])] for k in range(n)])
    return a_matrix


def __make_lambdas__(a_matrix, b_matrix, eps, method):
    return np.array([__minimize_equation__(a_matrix, b, eps, method) for b in b_matrix])


def __make_split_lambdas__(a_matrix, b_matrix, eps, method, dims_x_i, p):
    lambdas = []

    for i in range(len(b_matrix)):
        start = 0
        lambdas_i = []
        for end in accumulate(dims_x_i * p):
            lambdas_i.append(__minimize_equation__(a_matrix[:, start:end], b_matrix[i], eps, method))
            start = end
        lambdas.append(np.hstack(lambdas_i))

    return np.array(lambdas)


def __x_i_j_dimensions__(x_matrix):
    return [i for i in range(len(x_matrix)) for j in x_matrix[i]]


def __calculate_psi_line__(line, maper_x_i_j, p):
    psi_list = []
    cursor = 0
    for i in maper_x_i_j:
        psi_list.append(np.sum(line[cursor:cursor + p[i]]))
        cursor += p[i]
    return psi_list


def __make_psi__(a_matrix, x_matrix, lambdas, p):
    psi = []
    n = len(x_matrix[0][0])
    maper_x_i_j = [i for i in range(len(x_matrix)) for j in x_matrix[i]]

    for i in range(len(lambdas)):
        psi_i = []
        for q in range(n):
            psi_i.append(__calculate_psi_line__(a_matrix[q] * lambdas[i], maper_x_i_j, p))
        psi.append(psi_i)
    return np.array(psi)


def __make_a_small_matrix__(y_matrix, psi_matrix, eps, method, dims_x_i):
    a = []
    for i in range(len(y_matrix)):
        a_i_k = []
        last = 0
        for j in accumulate(dims_x_i, func=add):
            a_i_k.append(__minimize_equation__(psi_matrix[i][:, last:j], y_matrix[i], eps, method))
            last = j
        a.append(list(chain(a_i_k)))
    return np.array(a)


def __make_f_i__(a_small, psi_matrix, dims_x_i):
    f_i = []
    for i in range(len(a_small)):
        f_i_i = []
        last = 0
        for count, j in zip(range(len(dims_x_i)), accumulate(dims_x_i, func=add)):
            f_i_i.append(np.sum(a_small[i][count] * psi_matrix[i][:, last:j], axis=1))
            last = j
        f_i.append(f_i_i)
    return np.array(f_i)


def __make_c_small__(y_matrix, f_i, eps, method):
    return np.array([__minimize_equation__(np.column_stack(i), j, eps, method) for i, j in zip(f_i, y_matrix)])


def __make_f__(f_i, c):
    return np.array([np.sum(list(map(mul, i, j)), axis=0) for i, j in zip(f_i, c)])


def __real_f__(real_y, f):
    real_f = []
    for i, j in zip(real_y, f):
        i_min, i_max = np.min(i), np.max(i)
        real_f.append(j * (i_max - i_min) + i_min)
    return np.array(real_f)


def __plot_of_y_y_approximation__(y, y_approximation, name='no_name'):
    plt.title(name)
    plt.plot(y, 'b')
    plt.plot(y_approximation, 'r')
    plt.show()


def __show_plots__(y, y_approximation):
    for i, j, k in zip(y, y_approximation, range(len(y))):
        __plot_of_y_y_approximation__(i, j, 'Y-{:d}'.format(k + 1))


def process_calculations(data, degrees, weights, method, poly_type='chebyshev', find_split_lambdas=False, **kwargs):
    eps = CONST_EPS
    if 'epsilon' in kwargs:
        try:
            eps = DEFAULT_FLOAT_TYPE(kwargs['epsilon'])
        finally:
            pass

    x = deepcopy(data['x'])
    y = deepcopy(data['y'])

    dims_x_i = np.array([len(x[i]) for i in sorted(x)])

    x_matrix = np.array([x[i] for i in sorted(x)])
    y_matrix = np.array([y[i] for i in sorted(y)])

    # norm data
    x_normed_matrix, x_scales = __normalize_x_matrix__(x_matrix)
    y_normed_matrix, y_scales = __normalize_y_matrix__(y_matrix)

    p = np.array(degrees)
    weights = np.array(weights)
    polynom_type = __get_polynom_function__(poly_type)

    b_matrix = __make_b_matrix__(y_normed_matrix, weights)

    a_matrix = __make_a_matrix__(x_normed_matrix, p, polynom_type)

    if find_split_lambdas:
        lambdas = __make_split_lambdas__(a_matrix, b_matrix, eps, method, dims_x_i, p)
    else:
        lambdas = __make_lambdas__(a_matrix, b_matrix, eps, method)

    psi_matrix = __make_psi__(a_matrix, x_normed_matrix, lambdas, p)

    a_small = __make_a_small_matrix__(y_normed_matrix, psi_matrix, eps, method, dims_x_i)

    f_i = __make_f_i__(a_small, psi_matrix, dims_x_i)

    c = __make_c_small__(y_normed_matrix, f_i, eps, method)
    f = __make_f__(f_i, c)
    real_f = __real_f__(y_matrix, f)

    normed_error = np.linalg.norm(y_normed_matrix - f, np.inf, axis=1)
    error = np.linalg.norm(y_matrix - real_f, np.inf, axis=1)
    error = "normed Y errors - {:s}\nY errors - {:s}".format(str(normed_error), str(error))

    result = polynom_representation_add(polynom_type, p, dims_x_i, x_scales, lambdas, a_small, c)

    return "\n\n".join([result, error]), lambda: __show_plots__(y_matrix, real_f)


def __calculate_error_for_degrees__(degrees, x_normed_matrix, y_normed_matrix, y_matrix, b_matrix, dims_x_i,
                                    polynom_type, eps, method, find_split_lambdas):
    p = np.array(degrees)

    a_matrix = __make_a_matrix__(x_normed_matrix, p, polynom_type)
    if find_split_lambdas:
        lambdas = __make_split_lambdas__(a_matrix, b_matrix, eps, method, dims_x_i, p)
    else:
        lambdas = __make_lambdas__(a_matrix, b_matrix, eps, method)
    psi_matrix = __make_psi__(a_matrix, x_normed_matrix, lambdas, p)
    a_small = __make_a_small_matrix__(y_normed_matrix, psi_matrix, eps, method, dims_x_i)
    f_i = __make_f_i__(a_small, psi_matrix, dims_x_i)
    c = __make_c_small__(y_normed_matrix, f_i, eps, method)
    f = __make_f__(f_i, c)
    real_f = __real_f__(y_matrix, f)

    return {'norm': np.linalg.norm(y_normed_matrix - f, np.inf, axis=1), 'degrees': p, 'f': real_f}


def find_best_degrees(data, max_degrees, weights, method, poly_type='chebyshev', find_split_lambdas=False, **kwargs):
    results = []

    eps = CONST_EPS
    if 'epsilon' in kwargs:
        try:
            eps = DEFAULT_FLOAT_TYPE(kwargs['epsilon'])
        finally:
            pass

    x = data['x']
    y = data['y']

    dims_x_i = np.array([len(x[i]) for i in sorted(x)])

    x_matrix = np.array([x[i] for i in sorted(x)])
    y_matrix = np.array([y[i] for i in sorted(y)])

    # norm data
    x_normed_matrix, x_scales = __normalize_x_matrix__(x_matrix)
    y_normed_matrix, y_scales = __normalize_y_matrix__(y_matrix)

    max_p = np.array(max_degrees)
    weights = np.array(weights)
    polynom_type = __get_polynom_function__(poly_type)

    b_matrix = __make_b_matrix__(y_normed_matrix, weights)

    pool = mp.Pool()

    for current_degree in product(*[range(1, i) for i in max_p]):
        pool.apply_async(__calculate_error_for_degrees__, args=(
            current_degree, x_normed_matrix, y_normed_matrix, y_matrix, b_matrix, dims_x_i, polynom_type, eps, method,
            find_split_lambdas), callback=lambda result: results.append(result))
    pool.close()
    pool.join()

    best_results = []
    for i in range(len(y_normed_matrix)):
        res = min(results, key=lambda arg: arg['norm'][i])
        br = {'norm': res['norm'][i], 'degrees': res['degrees'], 'f': res['f'][i]}
        best_results.append(br)

    text_result = '\n'.join('Best degrees for Y{} are {} with normed error - {}'.format(i + 1,
                                                                                        __convert_degrees_to_string__(
                                                                                            best_results[i]['degrees']),
                                                                                        best_results[i]['norm']) for i
                            in range(len(best_results)))

    plots = lambda: __show_plots__(y_matrix, [br['f'] for br in best_results])

    return text_result, plots