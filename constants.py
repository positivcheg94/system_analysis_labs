import numpy

DEFAULT_FLOAT_TYPE = numpy.float

DEFAULT_FORM = 'add'
FORMS = ['add', 'mul', 'mul-add']

DEFAULT_METHOD = 'Least squares'
OPTIMIZATION_METHODS = {'Least squares': 'lstsq', 'Coordinate descent': 'cdesc', 'Seidel': 'seidel', 'Jakobi': 'jakobi',
                        'Conjugate gradient': 'conj'}
