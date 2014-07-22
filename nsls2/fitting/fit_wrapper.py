# Copyright (c) Brookhaven National Lab 2O14
# All rights reserved
# BSD License
# See LICENSE for full text


import numpy as np
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
import scipy.optimize
from fitting_tool.fitting_algorithm import leastsqbound

def test_func(p, x):

    a, b, c = p

    return a * np.exp(-b * x) + c


def residuals(p, y, x, weights):

    y0 = test_func(p, x)
    
    return (y0 - y)*weights


def fit(x, y, param_dict, target_function=None, fitting_engine='leastsq', 
        limit_dict=None, **engine_dict):
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
        Function object to compute residuals
        residuals = target_function(parameters, x, y, weights)

    fitting_engine : function_name
        i.e., leastsq

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
        Dictionary of limits for the param_dict. Keys must be the same as the
        param_dict

    engine_dict : dict
        Dictionary of keyword arguments that the specific fitting_engine
        requires

    """


    plist = []
    for item in param_dict.values():
        print item
        plist.append(item)

    if engine_dict.has_key('maxiter'):
        maxiter = engine_dict['maxiter']
    else:
        maxiter = 100
        
    if engine_dict.has_key('weights'):
        weights = engine_dict['weights']
    else:
        weights = np.ones(len(y))


    if fitting_engine == 'leastsq':
        output = leastsq_engine(x, y, plist, target_function=target_function, 
                                weights=weights, maxiter=maxiter)

    if fitting_eninge == 'leqstsqbound':
        lsb = leastsqbound()
        output = lsb.leastsqbound(target_function, plist, bounds, args=(x,y,weights), full_output=True)
        
    
    return output


def leastsq_engine(x, y, parameters, target_function=None,
                   weights=None, maxiter=100, full_output=True):
    """
    call scipy.optimize.leastsq function
    please refer to document of scipy.optimize.leastsq for detailed information
    call scipy.optimize.leastsq function
    One may refer to document of scipy.optimize.leastsq for detailed information

    Parameters
    ----------
    x : array-like
        x-coordinates

    y : array-like
        y-coordinates

    parameters : list
        List of parameters

    target_function : object
        Function object to compute residuals
        target_function(parameters, x, y, weights)
    
    Returns
    -------
    p1: list
        the solution of fitted parameters
    please refer to scipy.optimize.leastsq 
    for detailed information of other returns
    
    """

    p1, cov, infodict, mesg, success = scipy.optimize.leastsq(target_function, 
                                                              parameters, args=(y, x, weights),
                                                              maxfev=maxiter,
                                                              full_output = full_output)
    
    return p1, cov, infodict, mesg, success


def test():
    
    p = [2.5, 1.3, 0.5]
    xdata = np.linspace(0, 4, 50)
    ydata = test_func(p, xdata)

    #y = residuals(p, ydata, xdata)

    ydata = ydata + 0.2 * np.random.normal(size=len(xdata))

    plt.plot(xdata, ydata)
    plt.show()
    

    w = np.ones(len(ydata))
    #maxiter = 100
    
    p0 = [2.5, 1.0, 0.1]
    
    engine_dict={'maxiter': 100}
    
    param_dict = {'a':2.5, 'b': 1.3, 'c':0.5}
    
    #['a', 'b', 'c']
    
    p1 = fit(xdata, ydata, param_dict, target_function=residuals, weights=w, maxiter=50)
    
    ynew = test_func(p1, xdata)
    plt.plot(xdata, ydata, xdata, ynew)
    plt.show()
    
    res = residuals(p1, ydata, xdata, w)
    print np.sum(res)
    
    
    return
    


if __name__=="__main__":
    test()



