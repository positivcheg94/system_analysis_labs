from copy import deepcopy
import numpy as np
from functional_restoration.private.constants import DEFAULT_FLOAT_TYPE, CONST_EPS
from functional_restoration.private.shared import *
from functional_restoration.model.additive_model import Additive
from functional_restoration.model.multiplicative_model import Multiplicative

models_map = {'add': Additive, 'mul': Multiplicative}


class MixedResult:
    def __init__(self, model_results):
        self._model_results = model_results
        self._f_real = np.sum([i._f_real for i in self._model_results], axis=0)
        self._y_matrix = self._model_results[0]._y_matrix
        first_res = self._model_results[0]
        self._error = np.linalg.norm(self._y_matrix - self._f_real, np.inf, axis=1)
        self._normed_error = self._error * first_res.normed_error() / first_res.error()

    def plot(self):
        return show_plots(self._y_matrix, self._f_real)

    def text(self):
        return "Y normed errors - {:s}\nY errors - {:s}".format(str(self._normed_error), str(self._error))

    def predict(self, x_matrix, normalize=True):
        return np.sum([i.predict(x_matrix, normalize) for i in self._model_results], axis=0)


class Mixed:
    def __init__(self, degrees, weights, method, model_sequence, poly_type='chebyshev', find_split_lambdas=False,
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
        self._model_sequence = model_sequence
        self._poly_type = poly_type
        self._polynom_type = get_polynom_function(poly_type)
        self._find_split_lambdas = find_split_lambdas
        self._advanced_text_results = advanced_text_results

    def fit(self, x, y, normalize=True, independent=False):
        if independent:
            x_new = [[i] for i in x]
        else:
            x_new = x

        x_matrix = np.array(x_new)
        y_matrix = np.array(y)

        model_results = []

        y_last_residuals = y_matrix
        for i in self._model_sequence:
            current_result = models_map[i](self._p, self._weights, self._method, self._poly_type,
                                           self._find_split_lambdas).fit(x_matrix, y_last_residuals,
                                                                         normalize=normalize, independent=independent)
            y_last_residuals = y_last_residuals - current_result._f_real
            model_results.append(current_result)

        return MixedResult(model_results)
