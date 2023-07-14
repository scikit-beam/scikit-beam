from __future__ import absolute_import, division, print_function

import numpy as np


def fit_quad_to_peak(x, y):
    """
    Fits a quadratic to the data points handed in
    to the from y = b[0](x-b[1])**2 + b[2] and R2
    (measure of goodness of fit)

    Parameters
    ----------
    x : ndarray
        locations
    y : ndarray
        values

    Returns
    -------
    b : tuple
       coefficients of form y = b[0](x-b[1])**2 + b[2]

    R2 : float
      R2 value

    """

    lenx = len(x)

    # some sanity checks
    if lenx < 3:
        raise Exception("insufficient points handed in ")
    # set up fitting array
    X = np.vstack((x**2, x, np.ones(lenx))).T
    # use linear least squares fitting
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    SSerr = np.sum((np.polyval(beta, x) - y) ** 2)
    SStot = np.sum((y - np.mean(y)) ** 2)
    # re-map the returned value to match the form we want
    ret_beta = (beta[0], -beta[1] / (2 * beta[0]), beta[2] - beta[0] * (beta[1] / (2 * beta[0])) ** 2)

    return ret_beta, 1 - SSerr / SStot
