from copy import deepcopy
import numpy as np
from functional_restoration.private.constants import DEFAULT_FLOAT_TYPE, CONST_EPS
from functional_restoration.private.shared import *
from functional_restoration.model.additive import Additive
from functional_restoration.model.multiplicative import Multiplicative


class MulAddResult:
    def __init__(self, multiplicative_result, additive_result):
        self._multiplicative_result = multiplicative_result
        self._additive_result = additive_result
        self._f_real = self._multiplicative_result._f_real + self._additive_result._f_real
        self._y_matrix = self._multiplicative_result._y_matrix
        self._error = np.linalg.norm(self._y_matrix - self._f_real, np.inf, axis=1)

    def plot(self):
        return show_plots(self._y_matrix, self._f_real)

    def text(self):
        multiplicative_text = 'Multiplicative\n{}\n'.format(self._multiplicative_result.text())
        additive_text = 'Additive\n{}\n'.format(self._additive_result.text())
        return "Y errors - {:s}\n\n{:s}\n\n{:s}".format(str(self._error), multiplicative_text, additive_text)

    def predict(self, x_matrix):
        pass


class MulAdd:
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
        self._poly_type = poly_type
        self._polynom_type = get_polynom_function(poly_type)
        self._find_split_lambdas = find_split_lambdas
        self._advanced_text_results = advanced_text_results

    def fit(self, x, y):
        x_matrix = np.array(x)
        y_matrix = np.array(y)

        mul_res = Multiplicative(self._p, self._weights, self._method, self._poly_type, self._find_split_lambdas).fit(x,y)

        y_resid = y_matrix - mul_res._f_real

        add_res = Additive(self._p, self._weights, self._method, self._poly_type, self._find_split_lambdas).fit(x,y_resid)

        return MulAddResult(mul_res, add_res)
