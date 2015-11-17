from itertools import accumulate
from operator import add
from copy import deepcopy
import numpy as np
from numpy.polynomial import Polynomial, Chebyshev, Legendre, Laguerre, Hermite
from constants import *


def __make_psi_polynom__(lambdas, polynom, dims_x_i, p):
    def __make_basis__(degree):
        max_basis = []
        for deg in range(1, degree + 1):
            b = [0] * deg
            b[-1] = 1
            max_basis.append(polynom(b, domain=[0, 1], window=[0, 1]))
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
        psi_polynoms.append(np.vstack(psi_i_polynoms))
    return np.array(psi_polynoms)


def __make_f_i_polynoms__(psi_polynoms, a_small, dims_x_i):
    f_i = []
    for i in range(len(psi_polynoms)):
        f_i_j_groups = []
        last = 0
        bounds = list(accumulate(dims_x_i))

        for j in range(len(bounds)):
            f_i_j_groups.append(a_small[i][j] * psi_polynoms[i][last:last + bounds[j]])
            last = bounds[j]
        f_i.append(f_i_j_groups)
    return np.array(f_i)


def __make_f_polynoms__(f_i_polynoms, c):
    f_polynoms = deepcopy(f_i_polynoms)
    for i in range(len(f_polynoms)):
        for j in range(len(f_polynoms[i])):
            f_i_polynoms[i][j] = f_i_polynoms[i][j] * c[i][j]
    return f_polynoms


def __make_real_f_polynoms__(f_i_polynoms, x_scales):
    return 'lol'


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
        f_real = __make_real_f_polynoms__(f_polynoms, self.x_scales)
        print('lalala')