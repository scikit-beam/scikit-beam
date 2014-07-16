# Copyright (c) Brookhaven National Lab 2O14
# All rights reserved
# BSD License
# See LICENSE for full text



def fit(x, y, param_dict, fitting_engine, target_function, limit_dict=None,
        engine_dict=None):
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

    pass


