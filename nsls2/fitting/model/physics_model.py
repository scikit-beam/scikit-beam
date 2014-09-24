# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# @author: Li Li (lili@bnl.gov)                                        #
# created on 09/10/2014                                                #
#                                                                      #
# Original code:                                                       #
# @author: Mirna Lerotic, 2nd Look Consulting                          #
#         http://www.2ndlookconsulting.com/                            #
# Copyright (c) 2013, Stefan Vogt, Argonne National Laboratory         #
# All rights reserved.                                                 #
#                                                                      #
# Redistribution and use in source and binary forms, with or without   #
# modification, are permitted provided that the following conditions   #
# are met:                                                             #
#                                                                      #
# * Redistributions of source code must retain the above copyright     #
#   notice, this list of conditions and the following disclaimer.      #
#                                                                      #
# * Redistributions in binary form must reproduce the above copyright  #
#   notice this list of conditions and the following disclaimer in     #
#   the documentation and/or other materials provided with the         #
#   distribution.                                                      #
#                                                                      #
# * Neither the name of the Brookhaven Science Associates, Brookhaven  #
#   National Laboratory nor the names of its contributors may be used  #
#   to endorse or promote products derived from this software without  #
#   specific prior written permission.                                 #
#                                                                      #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS  #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT    #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS    #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE       #
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,           #
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR   #
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)   #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,  #
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OTHERWISE) ARISING   #
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE   #
# POSSIBILITY OF SUCH DAMAGE.                                          #
########################################################################

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import inspect

from nsls2.fitting.model.physics_peak import (elastic_peak, compton_peak,
                                              gauss_peak, lorentzian_peak,
                                              lorentzian_squared_peak)
from nsls2.fitting.base.parameter_data import get_para
from lmfit import Model


def set_default(model_name, func_name):
    """set values and bounds to Model parameters in lmfit

    Parameters
    ----------
    model_name : class object
        Model class object from lmfit
    func_name : function
        function name of physics peak
    """
    paras = inspect.getargspec(func_name)
    default_len = len(paras.defaults)

    # the first argument is independent variable, also ignored
    # default values are not considered for fitting in this function
    my_args = paras.args[1:]
    para_dict = get_para()

    for name in my_args:

        if name not in para_dict.keys():
            continue

        my_dict = para_dict[name]
        if my_dict['bound_type'] == 'none':
            model_name.set_param_hint(name, vary=True)
        elif my_dict['bound_type'] == 'fixed':
            model_name.set_param_hint(name, vary=False, value=my_dict['value'])
        elif my_dict['bound_type'] == 'lo':
            model_name.set_param_hint(name, value=my_dict['value'], vary=True,
                                      min=my_dict['min'])
        elif my_dict['bound_type'] == 'hi':
            model_name.set_param_hint(name, value=my_dict['value'], vary=True,
                                      max=my_dict['max'])
        elif my_dict['bound_type'] == 'lohi':
            model_name.set_param_hint(name, value=my_dict['value'], vary=True,
                                      min=my_dict['min'], max=my_dict['max'])
        else:
            raise TypeError("Boundary type {0} can't be used".format(my_dict['bound_type']))

def _gen_class_docs(func):
    return ("Wrap the {} function for fitting within lmfit framework\n".format(func.__name__) +
           func.__doc__)


class ElasticModel(Model):

    __doc__ = _gen_class_docs(elastic_peak)

    def __init__(self, *args, **kwargs):
        super(ElasticModel, self).__init__(elastic_peak, *args, **kwargs)
        set_default(self, elastic_peak)
        self.set_param_hint('epsilon', value=2.96, vary=False)


class ComptonModel(Model):

    __doc__ = _gen_class_docs(compton_peak)

    def __init__(self, *args, **kwargs):
        super(ComptonModel, self).__init__(compton_peak, *args, **kwargs)
        set_default(self, compton_peak)
        self.set_param_hint('epsilon', value=2.96, vary=False)
        self.set_param_hint('matrix', value=False, vary=False)


class GaussModel(Model):

    __doc__ = _gen_class_docs(gauss_peak)

    def __init__(self, *args, **kwargs):
        super(GaussModel, self).__init__(gauss_peak, *args, **kwargs)


class LorentzianModel(Model):

    __doc__ = _gen_class_docs(lorentzian_peak)

    def __init__(self, *args, **kwargs):
        super(LorentzianModel, self).__init__(lorentzian_peak, *args, **kwargs)


class Lorentzian2Model(Model):

    __doc__ = _gen_class_docs(lorentzian_peak)

    def __init__(self, *args, **kwargs):
        super(Lorentzian2Model, self).__init__(lorentzian_squared_peak, *args, **kwargs)


def set_range(model_name,
              parameter_name, parameter_value,
              parameter_vary, parameter_range):
    """
    set up fitting parameters in lmfit model

    Parameters
    ----------
    model_name : class object
        Model class object from lmfit
    parameter_name : str
    parameter_value : value
    parameter_vary : str
        fixed, free or bounded
    parameter_range : list
        [min, max]
    """
    if parameter_vary == 'fixed':
        model_name.set_param_hint(parameter_name, value=parameter_value, vary=False)
    elif parameter_vary == 'free':
        model_name.set_param_hint(parameter_name, value=parameter_value)
    elif parameter_vary == 'bounded':
        model_name.set_param_hint(parameter_name, value=parameter_value,
                                  min=parameter_range[0], max=parameter_range[1])
    else:
        raise ValueError("unrecognized value {0}".format(parameter_vary))


def gauss_fit(input_data,
              area, area_vary, area_range,
              center, center_vary, center_range,
              sigma, sigma_vary, sigma_range,):
    """
    wrapper of gaussian fitting model for vistrails.

    Parameters
    ----------
    input_data : array
        input data of x and y
    area : float
        area under peak profile
    area_vary : str
        fixed, free or bounded
    area_range : list
        bounded range
    center : float
        center position
    center_vary : str
        fixed, free or bounded
    center_range : list
        bounded range
    sigma : float
        standard deviation
    sigma_vary : str
        fixed, free or bounded
    sigma_range : list
        bounded range

    Returns
    -------
    param : dict
        fitting results
    x_data : array
        independent variable x
    y_data : array
        experimental data
    y_fit : array
        fitted y
    """

    x_data, y_data = input_data

    g = GaussModel()
    set_range(g, 'area', area, area_vary, area_range)
    set_range(g, 'center', center, center_vary, center_range)
    set_range(g, 'sigma', sigma, sigma_vary, sigma_range)

    result = g.fit(y_data, x=x_data)
    param = result.values
    y_fit = result.best_fit

    return param, x_data, y_data, y_fit


def lorentzian_fit(input_data,
                   area, area_vary, area_range,
                   center, center_vary, center_range,
                   sigma, sigma_vary, sigma_range,):
    """
    wrapper of lorentzian fitting model for vistrails.

    Parameters
    ----------
    input_data : array
        input data of x and y
    area : float
        area under peak profile
    area_vary : str
        fixed, free or bounded
    area_range : list
        bounded range
    center : float
        center position
    center_vary : str
        fixed, free or bounded
    center_range : list
        bounded range
    sigma : float
        standard deviation
    sigma_vary : str
        fixed, free or bounded
    sigma_range : list
        bounded range

    Returns
    -------
    param : dict
        fitting results
    x_data : array
        independent variable x
    y_data : array
        experimental data
    y_fit : array
        fitted y
    """

    x_data, y_data = input_data

    g = LorentzianModel()
    set_range(g, 'area', area, area_vary, area_range)
    set_range(g, 'center', center, center_vary, center_range)
    set_range(g, 'sigma', sigma, sigma_vary, sigma_range)

    result = g.fit(y_data, x=x_data)
    param = result.values
    y_fit = result.best_fit

    return param, x_data, y_data, y_fit


def lorentzian2_fit(input_data,
                    area, area_vary, area_range,
                    center, center_vary, center_range,
                    sigma, sigma_vary, sigma_range):
    """
    wrapper of lorentzian squared fitting model for vistrails.

    Parameters
    ----------
    input_data : array
        input data of x and y
    area : float
        area under peak profile
    area_vary : str
        fixed, free or bounded
    area_range : list
        bounded range
    center : float
        center position
    center_vary : str
        fixed, free or bounded
    center_range : list
        bounded range
    sigma : float
        standard deviation
    sigma_vary : str
        fixed, free or bounded
    sigma_range : list
        bounded range

    Returns
    -------
    param : dict
        fitting results
    x_data : array
        independent variable x
    y_data : array
        experimental data
    y_fit : array
        fitted y
    """

    x_data, y_data = input_data

    g = Lorentzian2Model()
    set_range(g, 'area', area, area_vary, area_range)
    set_range(g, 'center', center, center_vary, center_range)
    set_range(g, 'sigma', sigma, sigma_vary, sigma_range)

    result = g.fit(y_data, x=x_data)
    param = result.values
    y_fit = result.best_fit

    return param, x_data, y_data, y_fit

