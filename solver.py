from copy import deepcopy
from itertools import accumulate, chain
from operator import add, mul
from scipy import special
from scipy.optimize import minimize_scalar
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import Workbook

from constants import *
from polynom_representation import Representation


# shit funcs
def print_matrix_to_ws(ws, m, name):
    ws.append([name])
    for i in m:
        ws.append(i.tolist())


def __set_value_on_position__(array, position, value):
    array[position] = value
    return array


def __coord_descent__(a_matrix, b_vector, eps):
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


def __gauss_seidel__(a_matrix, b_vector, eps):
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


def __jacobi__(a_matrix, b_vector, eps):
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


def __minimize_equation__(a_matrix, b_vector, method=DEFAULT_METHOD):
    if method is 'cdesc':
        return __coord_descent__(a_matrix, b_vector, CONST_EPS)
    elif method is 'seidel':
        return __gauss_seidel__(a_matrix, b_vector, CONST_EPS)
    elif method is 'jacobi':
        return __gauss_seidel__(a_matrix, b_vector, CONST_EPS)
    else:
        return np.linalg.lstsq(a_matrix, b_vector)[0]



def __tricky_minimize__(a_matrix, b, trick=False, method=DEFAULT_METHOD):
    if trick:
        a_transpose = a_matrix.T
        return __minimize_equation__(a_transpose.dot(a_matrix), a_transpose.dot(b), method)
    return __minimize_equation__(a_matrix, b, method)


def __normalize_vector__(v):
    v_min, v_max = np.min(v), np.max(v)
    l = v_max - v_min
    scales = np.array([-v_min, l])
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


def __make_lambdas__(a_matrix, b_matrix):
    return np.array([__tricky_minimize__(a_matrix, b) for b in b_matrix])


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


def __make_a_small_matrix__(y_matrix, psi_matrix, dims_x_i):
    a = []
    for i in range(len(y_matrix)):
        a_i_k = []
        last = 0
        for j in accumulate(dims_x_i, func=add):
            a_i_k.append(__tricky_minimize__(psi_matrix[i][:, last:j], y_matrix[i]))
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


def __make_c_small__(y_matrix, f_i):
    return np.array([__tricky_minimize__(np.column_stack(i), j) for i, j in zip(f_i, y_matrix)])


def __make_f__(f_i, c):
    return np.array([np.sum(list(map(mul, i, j)), axis=0) for i, j in zip(f_i, c)])


def __real_f__(real_y, f):
    real_f = []
    for i, j in zip(real_y, f):
        i_min, i_max = np.min(i), np.max(i)
        real_f.append(j * (i_max - i_min) + i_min)
    return np.array(real_f)


class Solver(object):
    def __init__(self, data, samples, degrees, weights, poly_type='chebyshev', find_split_lambdas=False):
        self.wb = Workbook()
        self.ws = self.wb.active

        self.n = samples
        self.data = data
        self.x = deepcopy(self.data['x'])
        self.y = deepcopy(self.data['y'])

        self.dims_x_i = np.array([len(self.x[i]) for i in sorted(self.x)])

        self.X = np.array([self.x[i] for i in sorted(self.x)])
        self.Y = np.array([self.y[i] for i in sorted(self.y)])

        # norm data
        self.normed_X, self.x_scales = __normalize_x_matrix__(self.X)
        self.normed_Y, self.y_scales = __normalize_y_matrix__(self.Y)

        self.p = np.array(degrees)
        self.weights = weights
        self.polynom_type = __get_polynom_function__(poly_type)
        self.find_split_lambdas = find_split_lambdas

    def do_something(self):
        n = len(self.normed_X[0][0])
        real_y = self.Y
        x_normed = self.normed_X
        dims_x_i = self.dims_x_i
        y_normed = self.normed_Y
        p = self.p
        B = __make_b_matrix__(y_normed, self.weights)
        A = __make_a_matrix__(x_normed, p, self.polynom_type)
        lambdas = __make_lambdas__(A, B)
        psi_matrix = __make_psi__(A, x_normed, lambdas, p)
        a_small = __make_a_small_matrix__(y_normed, psi_matrix, dims_x_i)
        f_i = __make_f_i__(a_small, psi_matrix, dims_x_i)
        c = __make_c_small__(y_normed, f_i)
        f = __make_f__(f_i, c)
        real_f = __real_f__(real_y, f)

        arg = np.arange(n)

        plt.plot(arg, real_y[0], 'b', arg, real_f[0], 'r')
        plt.show()

        plt.plot(arg, real_y[1], 'b', arg, real_f[1], 'r')
        plt.show()

        repr = Representation(self.polynom_type, p,c,f_i, a_small, psi_matrix,lambdas,self.x_scales,self.dims_x_i, self.y_scales)
        repr.do_calculations()


