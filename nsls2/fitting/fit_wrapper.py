# Copyright (c) Brookhaven National Lab 2O14
# All rights reserved
# BSD License
# See LICENSE for full text

import numpy as np
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
import scipy.optimize


def target(x, y, **args):

    a = args["a"]
    b = args["b"]
    c = args["c"]

    return a * np.exp(-b * x) + c -y


def fit(x, y, fitting_engine=None, target_function=None, limit_dict=None, 
        **param_dict):
    """
    Top-level function for fitting, magic and ponies

    Parameters
    ----------
    x : array-like
        x-coordinates

    y : array-like
        y-coordinates

    param_dict : dict
        Dictionary of parameters

    target_function : object
        Function object that is used to compute and compare against
        y = target_function(x, **param_dict)
            x is the array of all of your data (the same parameter that was
            passed in to this function)

    fitting_engine : function_name
        Something that allows you to modify parameters to 'fit' the
        target_function to the (x,y) data
        fit_param_dict = function_name(x, y, target_function, param_dict,
                                       constraint_dict, engine_dict)

    Returns
    -------
    fit_param_dict : dict
        Dictionary of the fit values of the parameters.  Keys are the same as
        the param_dict and the limit_dict

    correlation_matrix : pandas.DataFrame
        Table of correlations (named rows/cols)

    covariance_matrix : pandas.DataFrame
        Table of covariance (named rows/cols)

    residuals : np.ndarray
        Returned as (data - fit)

    Optional
    --------
    limit_dict : dict
        Dictionary of limits for the param_dict.  Keys must be the same as the
        param_dict

    engine_dict : dict
        Dictionary of keyword arguments that the specific fitting_engine
        requires

    """

    # calls the engine

    # compute covariance

    # compute correlation

    # compute residuals

    # returns

    maxiter = 100
    weights = np.ones(len(y))
    
    errfunc = target_function(x, y, **param_dict)

    p0 = []
    for item in param_dict.items():
        p0.append(item)
        
    print p0

    p1, cov, infodict, mesg, success = scipy.optimize.leastsq(errfunc, 
                                                              p0, args=(y, x, weights),
                                                              maxfev=maxiter, full_output = True)

    

    return



def test():
    xdata = np.linspace(0, 4, 50)
    ydata = xdata

    y = target(xdata, xdata, a=2.5, b=1.3, c=0.5)

    ydata = y + 0.2 * np.random.normal(size=len(xdata))

    plt.plot(xdata, ydata)
    plt.show()
    
    print ydata
    
    fit(xdata, ydata, target_function=target, a=2.5, b=1.3, c=0.5)
    
    return


if __name__=="__main__":
    test()











