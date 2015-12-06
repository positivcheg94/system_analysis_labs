from itertools import accumulate
from operator import add

import numpy as np
from numpy.polynomial import Polynomial

from functional_restoration.representation.shared import *
from functional_restoration.private.constants import DOM


def make_psi_polynom(lambdas, polynom, dims_x_i, p):
    basis = make_basis(polynom, np.max(p))

    stages = np.roll(list(accumulate(dims_x_i * p, func=add)), 1)
    stages[0] = 0

    psi_polynoms = []
    for q in range(len(lambdas)):
        psi_i_polynoms = []
        for i in range(len(stages)):
            psi_i_j_polynoms = []
            for j in range(dims_x_i[i]):
                start = stages[i] + j * p[i]
                end = start + p[i]
                psi_i_j_polynoms.append(np.sum(lambdas[q][start:end] * basis[:p[i]]))
            psi_i_polynoms.append(psi_i_j_polynoms)
        psi_polynoms.append(psi_i_polynoms)
    return np.array(psi_polynoms)


def make_f_i_polynoms(psi_polynoms, a_small):
    f_i = []
    for i in range(len(psi_polynoms)):
        f_i_j_groups = []
        for j in range(len(psi_polynoms[i])):
            f_i_j_groups.append(a_small[i][j] * psi_polynoms[i][j])
        f_i.append(f_i_j_groups)
    return np.array(f_i)


def make_f_polynoms(f_i_polynoms, c):
    f_polynoms = []
    for i in range(len(f_i_polynoms)):
        f_i = []
        for j in range(len(f_i_polynoms[i])):
            f_i.append(f_i_polynoms[i][j] * c[i][j])
        f_polynoms.append(f_i)
    return np.array(f_polynoms)


def transform_f_to_usual_polynomial_form(f_polynoms):
    real_polynom = [[[j.convert(kind=Polynomial, domain=DOM, window=DOM) for j in i] for i in p] for p in f_polynoms]
    return np.array(real_polynom)


def make_unshifted_f_real_polynoms(f_real_polynoms, x_scales):
    p_x_0 = Polynomial([0, 1])
    unshifted_f_real_polynom = []
    for q in range(len(f_real_polynoms)):
        f_i = []
        for i in range(len(f_real_polynoms[q])):
            f_i_g = []
            for j in range(len(f_real_polynoms[q][i])):
                x_shift, x_zoom = x_scales[i][j]
                f_i_g.append(f_real_polynoms[q][i][j]((p_x_0 - x_shift) / x_zoom))
            f_i.append(f_i_g)
        unshifted_f_real_polynom.append(f_i)
    return np.array(unshifted_f_real_polynom)


def __psi_representation__(psi, symbol):
    psi_repr = []
    for i in range(len(psi)):
        for x_i in range(len(psi[i])):
            for x_i_j in range(len(psi[i][x_i])):
                psi_repr.append('PSI-{:d},{:d} = '.format(x_i + 1, x_i_j + 1) + convert_special_polynom_to_string(
                    psi[i][x_i][x_i_j], x_i + 1, x_i_j + 1, symbol))
        return '\n\n'.join(psi_repr)


def __f_i_representation__(f_i_polynom, symbol):
    f_i_repr = []
    f = f_i_polynom
    for i in range(len(f)):
        for x_i in range(len(f[i])):
            f_i_j_repr = []
            for x_i_j in range(len(f[i][x_i])):
                f_i_j_repr.append(convert_special_polynom_to_string(f[i][x_i][x_i_j], x_i + 1, x_i_j + 1, symbol))
            f_i_repr.append('Fi-{:d},{:d} = '.format(x_i + 1, i + 1) + ' + '.join(f_i_j_repr))
    return '\n\n'.join(f_i_repr)


def __f_representation__(f_polynom, symbol):
    f_repr = []
    f = f_polynom
    for i in range(len(f)):
        f_i_repr = []
        for x_i in range(len(f[i])):
            for x_i_j in range(len(f[i][x_i])):
                f_i_repr.append(convert_special_polynom_to_string(f[i][x_i][x_i_j], x_i + 1, x_i_j + 1, symbol))
        f_repr.append('F-{:d} = '.format(i + 1) + ' + '.join(f_i_repr))
    return '\n\n'.join(f_repr)


def __f_general_polynom_representation__(f_real):
    f_repr = []
    f = f_real
    for i in range(len(f)):
        f_i_repr = []
        for x_i in range(len(f[i])):
            for x_i_j in range(len(f[i][x_i])):
                f_i_repr.append(convert_polynom_to_string(f[i][x_i][x_i_j], x_i + 1, x_i_j + 1))
        f_repr.append('F-{:d} = '.format(i + 1) + ' + '.join(f_i_repr))
    return '\n\n'.join(f_repr)


def representation(polynom_type, p, dims_x_i, x_scales, lambdas, a_small, c):
    polynom, polynom_symbol = polynom_picker(polynom_type)

    psi_polynoms = make_psi_polynom(lambdas, polynom, dims_x_i, p)
    f_i_polynoms = make_f_i_polynoms(psi_polynoms, a_small)
    f_polynoms = make_f_polynoms(f_i_polynoms, c)
    f_real = transform_f_to_usual_polynomial_form(f_polynoms)
    unshifted_f_real = make_unshifted_f_real_polynoms(f_real, x_scales)

    psi_representation = __psi_representation__(psi_polynoms, polynom_symbol)
    f_i_representation = __f_i_representation__(f_i_polynoms, polynom_symbol)
    f_representation = __f_representation__(f_polynoms, polynom_symbol)
    f_real_representation = __f_general_polynom_representation__(f_real)
    unshifted_f_real_representation = __f_general_polynom_representation__(unshifted_f_real)

    text_representation = '\n\n\n\n'.join(
        [psi_representation, f_i_representation, f_representation, 'General form\n' + f_real_representation,
         'Unshifted form\n' + unshifted_f_real_representation])

    return f_polynoms, text_representation
