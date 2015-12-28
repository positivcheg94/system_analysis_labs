from statsmodels.tsa.filters import hp_filter
from statsmodels.tsa.ar_model import AR


def predict(x, prediction_length):
    start = n = len(x)
    end = start + prediction_length - 1
    trend, cycle = hp_filter.hpfilter(x)
    trend_prediction = AR(trend).fit().predict(start, end)
    cycle_prediction = AR(cycle).fit().predict(start, end)
    return trend_prediction + cycle_prediction


def bulk_predict(x_matrix, prediction_length):
    return [predict(i, prediction_length) for i in x_matrix]


def calculate_risk(y, y_abnormal, y_crash):
    if y > y_abnormal:
        return 0
    elif y > y_crash:
        return (y - y_abnormal) / (y_crash - y_abnormal)
    else:
        return 1
