import numpy

DEFAULT_FLOAT_TYPE = numpy.float128

CONST_WINDOW_SIZE = 1e-2
CONST_EPS = 1e-8

CHEBYSHEV = 'chebyshev'
CHEBYSHEV_SYMBOL = 'T'
LEGENDRE = 'legendre'
LEGENDRE_SYMBOL = 'P'
LAGUERRE = 'laguerre'
LAGUERRE_SYMBOL = 'L'
HERMITE = 'hermite'
HERMITE_SYMBOL = 'H'

NAME = 'temp.xlsx'

DEFAULT_METHOD = 'Least squares'

OPTIMIZATION_METHODS = {'Least squares': 'lstsq', 'Coordinate descent': 'cdesc', 'Seidel': 'seidel', 'Jakobi': 'jakobi',
                        'Conjugate gradient': 'conj'}
