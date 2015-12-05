from copy import deepcopy
from itertools import product
import multiprocessing as mp

import numpy as np

from functional_restoration.private.constants import DEFAULT_FLOAT_TYPE, CONST_EPS, CONST_EPS_LOG
from functional_restoration.private.shared import *
from functional_restoration.representation.multiplicative import representation


def make_a_matrix(x, p, polynom):
    n = len(x[0][0])
    a_matrix = np.array([[np.log(polynom(p_j, j[k]) + 1 + CONST_EPS_LOG) for x_i in range(len(x)) for j in x[x_i] for
                          p_j in range(p[x_i])] for k in range(n)])
    return a_matrix


def __calculate_error_for_degrees__(degrees, x_normed_matrix, y_normed_matrix, ln_y_add_one, y_scales, b_matrix,
                                    dims_x_i,
                                    polynom_type, eps, method, find_split_lambdas):
    p = np.array(degrees)

    a_matrix = make_a_matrix(x_normed_matrix, p, polynom_type)
    if find_split_lambdas:
        lambdas = make_split_lambdas(a_matrix, b_matrix, eps, method, dims_x_i, p)
    else:
        lambdas = make_lambdas(a_matrix, b_matrix, eps, method)
    psi_matrix = make_psi(a_matrix, x_normed_matrix, lambdas, p)
    a_small = make_a_small_matrix(ln_y_add_one, psi_matrix, eps, method, dims_x_i)
    f_i = make_f_i(a_small, psi_matrix, dims_x_i)
    c = make_c_small(ln_y_add_one, f_i, eps, method)
    f_log = make_f(f_i, c)
    f = np.exp(f_log) - 1
    f_real = make_real_f(y_scales, f)

    return {'norm': np.linalg.norm(y_normed_matrix - f, np.inf, axis=1), 'degrees': p, 'f': f_real}


class MultiplicativeResult:
    def __init__(self, dims_x_i, x_scales, y_scales, y_matrix, f_real, f_polynoms, lambdas, a_small, c, normed_error,
                 text_result):
        self._dims_x_i = dims_x_i
        self._x_scales = x_scales
        self._y_scales = y_scales
        self._y_matrix = y_matrix
        self._f_real = f_real
        self._f_polynoms = f_polynoms
        self._lambdas = lambdas
        self._a_small = a_small
        self._c = c
        self._normed_error = normed_error
        self._text_result = text_result

    def dims_x(self):
        return self._dims_x_i

    def lambdas(self):
        return self._lambdas

    def a_small(self):
        return self._a_small

    def c(self):
        return self._c

    def normed_error(self):
        return self._normed_error

    def text(self):
        return self._text_result

    def plot(self):
        return show_plots(self._y_matrix, self._f_real)

    def predict(self, x_matrix):
        pass


class Multiplicative:
    def __init__(self, degrees, weights, method, poly_type='chebyshev', find_split_lambdas=False,
                 advanced_text_results=True, **kwargs):
        self._eps = CONST_EPS
        if 'epsilon' in kwargs:
            try:
                self._eps = DEFAULT_FLOAT_TYPE(kwargs['epsilon'])
            finally:
                pass

        self._p = np.array(degrees)
        self._weights = np.array(weights)
        self._method = method
        self._polynom_type = get_polynom_function(poly_type)
        self._find_split_lambdas = find_split_lambdas
        self._advanced_text_results = advanced_text_results

    def fit(self, data):
        x = deepcopy(data['x'])
        y = deepcopy(data['y'])

        dims_x_i = np.array([len(x[i]) for i in sorted(x)])

        x_matrix = np.array([x[i] for i in sorted(x)])
        y_matrix = np.array([y[i] for i in sorted(y)])

        # norm data
        x_normed_matrix, x_scales = normalize_x_matrix(x_matrix)
        y_normed_matrix, y_scales = normalize_y_matrix(y_matrix)

        ln_y_add_one = np.log(y_normed_matrix + 1)

        b_matrix = make_b_matrix(ln_y_add_one, self._weights)
        a_matrix = make_a_matrix(x_normed_matrix, self._p, self._polynom_type)
        if self._find_split_lambdas:
            lambdas = make_split_lambdas(a_matrix, b_matrix, self._eps, self._method, dims_x_i, self._p)
        else:
            lambdas = make_lambdas(a_matrix, b_matrix, self._eps, self._method)
        psi_matrix = make_psi(a_matrix, x_normed_matrix, lambdas, self._p)
        a_small = make_a_small_matrix(ln_y_add_one, psi_matrix, self._eps, self._method, dims_x_i)
        f_i = make_f_i(a_small, psi_matrix, dims_x_i)
        c = make_c_small(ln_y_add_one, f_i, self._eps, self._method)
        f_log = make_f(f_i, c)
        f = np.exp(f_log) - 1
        f_real = make_real_f(y_scales, f)

        normed_error = np.linalg.norm(y_normed_matrix - f, np.inf, axis=1)

        error = np.linalg.norm(y_matrix - f_real, np.inf, axis=1)
        error_text = "normed Y errors - {:s}\nY errors - {:s}".format(str(normed_error), str(error))

        result_text = representation(self._polynom_type, self._p, dims_x_i, x_scales, lambdas, a_small, c)

        text = '\n\n'.join([error_text, result_text])

        return MultiplicativeResult(dims_x_i, x_scales, y_scales, y_matrix, f_real, None, lambdas, a_small, c,
                                    normed_error, text)


class MultiplicativeDegreeFinderResult:
    def __init__(self, dims_x_i, y_matrix, f_real, degrees, normed_error, text_result):
        self._dims_x_i = dims_x_i
        self._y_matrix = y_matrix
        self._f_real = f_real
        self._degrees = degrees
        self._normed_error = normed_error
        self._text_result = text_result

    def dims_x(self):
        return self._dims_x_i

    def degrees(self):
        return self._degrees

    def normed_error(self):
        return self._normed_error

    def text(self):
        return self._text_result

    def plot(self):
        return show_plots(self._y_matrix, self._f_real)


class MultiplicativeDegreeFinder:
    def __init__(self, max_degrees, weights, method, poly_type='chebyshev', find_split_lambdas=False, **kwargs):
        self._eps = CONST_EPS
        if 'epsilon' in kwargs:
            try:
                self._eps = DEFAULT_FLOAT_TYPE(kwargs['epsilon'])
            finally:
                pass

        self._max_p = np.array(max_degrees)
        self._weights = np.array(weights)
        self._method = method
        self._polynom_type = get_polynom_function(poly_type)
        self._find_split_lambdas = find_split_lambdas

    def fit(self, data):
        x = deepcopy(data['x'])
        y = deepcopy(data['y'])

        x_matrix = np.array([x[i] for i in sorted(x)])
        y_matrix = np.array([y[i] for i in sorted(y)])

        dims_x_i = np.array([len(x[i]) for i in sorted(x)])

        # norm data
        x_normed_matrix, x_scales = normalize_x_matrix(x_matrix)
        y_normed_matrix, y_scales = normalize_y_matrix(y_matrix)
        ln_y_add_one = np.log(y_normed_matrix + 1)

        b_matrix = make_b_matrix(y_normed_matrix, self._weights)

        pool = mp.Pool()
        results = []

        for current_degree in product(*[range(1, i) for i in self._max_p]):
            pool.apply_async(__calculate_error_for_degrees__, args=(
                current_degree, x_normed_matrix, y_normed_matrix, ln_y_add_one, y_scales, b_matrix, dims_x_i,
                self._polynom_type, self._eps, self._method, self._find_split_lambdas),
                             callback=lambda result: results.append(result))
        pool.close()
        pool.join()

        best_results = []
        for i in range(len(y_normed_matrix)):
            res = min(results, key=lambda arg: arg['norm'][i])
            br = {'norm': res['norm'][i], 'degrees': res['degrees'], 'f': res['f'][i]}
            best_results.append(br)

        f_real = [br['f'] for br in best_results]
        degrees = [br['degrees'] for br in best_results]
        normed_error = [br['norm'] for br in best_results]

        text_result = '\n'.join('Best degrees for Y{} are {} with normed error - {}'.format(i + 1,
                                                                                            convert_degrees_to_string(
                                                                                                best_results[i][
                                                                                                    'degrees']),
                                                                                            best_results[i]['norm']) for
                                i in range(len(best_results)))

        return MultiplicativeDegreeFinderResult(dims_x_i, y_matrix, f_real, degrees, normed_error, text_result)
