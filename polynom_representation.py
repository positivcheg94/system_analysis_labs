from itertools import accumulate, chain
from operator import add
from copy import deepcopy
import numpy as np
from numpy.polynomial import Polynomial, Chebyshev, Legendre, Laguerre, Hermite
from constants import *

DOM = np.array([0, 1])
POLYNOM_MASK = '{:e}*{:s}[{:d},{:d}]^{:d}'
SPECIAL_POLYNOM_MASK = '{:e}*{:s}{:d}[x{:d},{:d}]'


def __convert_polynom_to_string__(polynom, first, second, symbol='x'):
    coef = polynom.coef
    return ' + '.join([POLYNOM_MASK.format(coef[i], symbol, first, second, i) for i in reversed(range(len(coef)))])


def __convert_special_polynom_to_string__(polynom, first, second, symbol='C'):
    coef = polynom.coef
    return ' + '.join(
        [SPECIAL_POLYNOM_MASK.format(coef[i], symbol, i, first, second) for i in reversed(range(len(coef)))])


def __make_psi_polynom__(lambdas, polynom, dims_x_i, p):
    def __make_basis__(degree):
        max_basis = []
        for deg in range(1, degree + 1):
            b = [0] * deg
            b[-1] = 1
            max_basis.append(polynom(b, domain=DOM, window=DOM))
        return np.array(max_basis)

    basis = __make_basis__(np.max(p))

    stages = np.roll(list(accumulate(dims_x_i * p, func=add)), 1)
    stages[0] = 0

    psi_polynoms = []
    for q in range(len(lambdas)):
        psi_i_polynoms = []
        for i in range(len(stages)):
            for j in range(dims_x_i[i]):
                start = stages[i] + j * p[i]
                end = start + p[i]
                psi_i_polynoms.append(np.sum(lambdas[q][start:end] * basis[:p[i]]))
        psi_polynoms.append(psi_i_polynoms)
    return np.array(psi_polynoms)


def __make_f_i_polynoms__(psi_polynoms, a_small, dims_x_i):
    f_i = []
    for i in range(len(psi_polynoms)):
        f_i_j_groups = []
        last = 0
        bounds = list(accumulate(dims_x_i))

        for j in range(len(bounds)):
            f_i_j_groups.append(a_small[i][j] * psi_polynoms[i][last:bounds[j]])
            last = bounds[j]
        f_i.append(f_i_j_groups)
    return np.array(f_i)


def __make_f_polynoms__(f_i_polynoms, c):
    f_polynoms = deepcopy(f_i_polynoms)
    for i in range(len(f_polynoms)):
        for j in range(len(f_polynoms[i])):
            f_polynoms[i][j] = f_i_polynoms[i][j] * c[i][j]
    return f_polynoms


def __transform_f_to_usual_polynomial_form__(f_polynoms):
    real_polynom = [[[j.convert(kind=Polynomial, domain=DOM, window=DOM) for j in i] for i in p] for p in f_polynoms]
    return np.array(real_polynom)


def __unshifted_f_polynoms__(f_real_polynoms, x_scales):
    unshifted_real_polynom = []
    for f_polynom in f_real_polynoms:
        f_i = []
        for i in range(len(f_polynom)):
            f_i_g = []
            for j in range(len(f_polynom[i])):
                shift, zoom = x_scales[i][j]
                poly = Polynomial(f_polynom[i][j].convert(domain=DOM, window=DOM * zoom + shift).coef, domain=DOM,
                                  window=DOM)
                f_i_g.append(poly)
            f_i.append(f_i_g)
        unshifted_real_polynom.append(f_i)
    return np.array(unshifted_real_polynom)


def __psi_representation__(psi, dims_x_i, symbol):
    psi_repr = []
    for i in range(len(psi)):
        last = 0
        for x_i, k in zip(range(len(dims_x_i)), list(accumulate(dims_x_i))):
            for t in range(k - last):
                psi_repr.append(
                    'PSI-{:d},{:d} = '.format(x_i + 1, t + 1) + __convert_special_polynom_to_string__(psi[i][last + t],
                                                                                                      x_i + 1, t + 1,
                                                                                                      symbol))
            last = k
        return '\n\n'.join(psi_repr)


def __f_i_representation__(f_i_polynom, symbol):
    f_i_repr = []
    f = f_i_polynom
    for i in range(len(f)):
        for x_i in range(len(f[i])):
            f_i_j_repr = []
            for x_i_j in range(len(f[i][x_i])):
                f_i_j_repr.append(__convert_special_polynom_to_string__(f[i][x_i][x_i_j], x_i + 1, x_i_j + 1, symbol))
            f_i_repr.append('Fi-{:d},{:d} = '.format(x_i + 1, i + 1) + ' + '.join(f_i_j_repr))
    return '\n\n'.join(f_i_repr)


def __f_representation__(f_polynom, symbol):
    f_repr = []
    f = f_polynom
    for i in range(len(f)):
        f_i_repr = []
        for x_i in range(len(f[i])):
            for x_i_j in range(len(f[i][x_i])):
                f_i_repr.append(__convert_special_polynom_to_string__(f[i][x_i][x_i_j], x_i + 1, x_i_j + 1, symbol))
        f_repr.append('F-{:d} = '.format(i + 1) + ' + '.join(f_i_repr))
    return '\n\n'.join(f_repr)


def __f_general_polynom_representation__(f_real):
    f_repr = []
    f = f_real
    for i in range(len(f)):
        f_i_repr = []
        for x_i in range(len(f[i])):
            for x_i_j in range(len(f[i][x_i])):
                f_i_repr.append(__convert_polynom_to_string__(f[i][x_i][x_i_j], x_i + 1, x_i_j + 1))
        f_repr.append('F-{:d} = '.format(i + 1) + ' + '.join(f_i_repr))
    return '\n\n'.join(f_repr)


class Representation(object):
    @staticmethod
    def __polynom_picker__(polynom_type):
        if polynom_type is LEGENDRE:
            return Legendre, LEGENDRE_SYMBOL
        elif polynom_type is LAGUERRE:
            return Laguerre, LAGUERRE_SYMBOL
        elif polynom_type is HERMITE:
            return Hermite, HERMITE_SYMBOL
        else:
            return Chebyshev, CHEBYSHEV_SYMBOL

    def __init__(self, polynom_type, p, c, f_i, a_small, psi, lambdas, x_scales, dims_x_i, y_scales):
        self.polynom, self.polynom_symbol = Representation.__polynom_picker__(polynom_type)
        self.p = p
        self.c = np.array(c)
        self.f_i = np.array(f_i)
        self.a_small = np.array(a_small)
        self.psi = np.array(psi)
        self.lambdas = np.array(lambdas)
        self.x_scales = x_scales
        self.y_scales = y_scales
        self.dims_x_i = dims_x_i

    def do_calculations(self):
        dims_x_i = self.dims_x_i
        p = self.p
        psi_polynoms = __make_psi_polynom__(self.lambdas, self.polynom, dims_x_i, p)
        f_i_polynoms = __make_f_i_polynoms__(psi_polynoms, self.a_small, dims_x_i)
        f_polynoms = __make_f_polynoms__(f_i_polynoms, self.c)
        f_real = __transform_f_to_usual_polynomial_form__(f_polynoms)
        unshifted_f = __unshifted_f_polynoms__(f_real, self.x_scales)

        psi_representation = __psi_representation__(psi_polynoms, dims_x_i, self.polynom_symbol)
        f_i_representation = __f_i_representation__(f_i_polynoms, self.polynom_symbol)
        f_representation = __f_representation__(f_polynoms, self.polynom_symbol)
        f_real_representation = __f_general_polynom_representation__(f_real)
        unshifted_f_representation = __f_general_polynom_representation__(unshifted_f)
        return '\n\n\n\n'.join(
            [psi_representation, f_i_representation, f_representation, 'General form\n' + f_real_representation,
             'Unshifted form\n' + unshifted_f_representation])
