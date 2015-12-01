from copy import deepcopy
from itertools import accumulate
from operator import add
import numpy as np
from numpy.polynomial import Polynomial
from solver.representation.shared import *


def make_basis(polynom, degree):
    max_basis = []
    for deg in range(1, degree + 1):
        b = [0] * deg
        b[-1] = 1
        max_basis.append(polynom(b))
    return np.array(max_basis)


def make_psi_i_j(lambdas, basis, shift, zoom, i, j, symbol='x'):
    real_basis = [i((Polynomial([0, 1]) - shift) / zoom) for i in basis]
    return 'PSI[{},{}] = {} - 1'.format(i, j, ' * '.join(
        ['( {} )^( {} )'.format(convert_polynom_to_string(real_b, i, j), lambd) for real_b, lambd in
         zip(real_basis, lambdas)]))


def make_psi(lambdas, polynom, dims_x_i, p, x_scales):
    basis = make_basis(polynom, np.max(p))

    stages = np.roll(list(accumulate(dims_x_i * p, func=add)), 1)
    stages[0] = 0

    psi_polynoms = []
    for q in range(len(lambdas)):
        psi_i_polynoms = []
        for i in range(len(stages)):
            for j in range(dims_x_i[i]):
                start = stages[i] + j * p[i]
                end = start + p[i]
                shift, zoom = x_scales[i][j]
                psi_i_polynoms.append(make_psi_i_j(lambdas[q][start:end], basis[:p[i]], shift, zoom, i + 1, j + 1))
        psi_polynoms.append('\n\n'.join(psi_i_polynoms))
    return '\n\n'.join(psi_polynoms)


def make_f_i(a_small, dims_x_i):
    f_i = []
    for i in range(len(a_small)):
        f_i_j = []

        for j in range(len(dims_x_i)):
            tmp = ' * '.join(
                ['( 1 + PSI[{},{}] )^( {} )'.format(j + 1, k + 1, a_small[i][j][k]) for k in range(len(a_small[i][j]))])
            f_i_j.append('F[{},{}] = {} - 1'.format(i + 1, j + 1, tmp))
        f_i.append('\n\n'.join(f_i_j))
    return '\n\n'.join(f_i)


def make_f(c):
    f_i = []
    for i in range(len(c)):
        tmp = ' * '.join(['( 1 + F[{},{}] )^( {} )'.format(i+1,j+1, c[i][j]) for j in range(len(c[i]))])
        f_i.append('F[{}] = {} - 1'.format(i+1,tmp))
    return '\n\n'.join(f_i)


def representation(polynom_type, p, dims_x_i, x_scales, lambdas, a_small, c):
    polynom, polynom_symbol = polynom_picker(polynom_type)

    psi = make_psi(lambdas, polynom, dims_x_i, p, x_scales)
    f_i = make_f_i(a_small, dims_x_i)
    f = make_f(c)

    return '\n\n'.join([psi,f_i, f])
