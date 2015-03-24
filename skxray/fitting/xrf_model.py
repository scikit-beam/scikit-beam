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

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
from scipy.optimize import nnls
import six
import copy
from collections import OrderedDict

from skxray.constants.api import XrfElement as Element
from skxray.fitting.lineshapes import gaussian
from skxray.fitting.models import (ComptonModel, ElasticModel,
                                   _gen_class_docs)
from skxray.fitting.base.parameter_data import get_para
import skxray.fitting.base.parameter_data as sfb_pd
from skxray.fitting.background import snip_method
from lmfit import Model

import logging
logger = logging.getLogger(__name__)


# emission line energy between (1, 30) keV
# TODO Add _K after each of these.
K_LINE = ['Na_K', 'Mg_K', 'Al_K', 'Si_K', 'P_K', 'S_K', 'Cl_K', 'Ar_K', 'K_K',
    'Ca_K', 'Sc_K', 'Ti_K', 'V_K', 'Cr_K', 'Mn_K', 'Fe_K', 'Co_K', 'Ni_K',
    'Cu_K', 'Zn_K', 'Ga_K', 'Ge_K', 'As_K', 'Se_K', 'Br_K', 'Kr_K',
    'Rb_K', 'Sr_K', 'Y_K', 'Zr_K', 'Nb_K', 'Mo_K', 'Tc_K', 'Ru_K', 'Rh_K',
    'Pd_K', 'Ag_K', 'Cd_K', 'In_K', 'Sn_K', 'Sb_K', 'Te_K', 'I_K']

L_LINE = ['Ga_L', 'Ge_L', 'As_L', 'Se_L', 'Br_L', 'Kr_L', 'Rb_L', 'Sr_L', 'Y_L', 'Zr_L', 'Nb_L',
          'Mo_L', 'Tc_L', 'Ru_L', 'Rh_L', 'Pd_L', 'Ag_L', 'Cd_L', 'In_L', 'Sn_L', 'Sb_L', 'Te_L',
          'I_L', 'Xe_L', 'Cs_L', 'Ba_L', 'La_L', 'Ce_L', 'Pr_L', 'Nd_L', 'Pm_L', 'Sm_L', 'Eu_L',
          'Gd_L', 'Tb_L', 'Dy_L', 'Ho_L', 'Er_L', 'Tm_L', 'Yb_L', 'Lu_L', 'Hf_L', 'Ta_L', 'W_L',
          'Re_L', 'Os_L', 'Ir_L', 'Pt_L', 'Au_L', 'Hg_L', 'Tl_L', 'Pb_L', 'Bi_L', 'Po_L', 'At_L',
          'Rn_L', 'Fr_L', 'Ac_L', 'Th_L', 'Pa_L', 'U_L', 'Np_L', 'Pu_L', 'Am_L']

M_LINE = ['Hf_M', 'Ta_M', 'W_M', 'Re_M', 'Os_M', 'Ir_M', 'Pt_M', 'Au_M', 'Hg_M', 'TL_M', 'Pb_M', 'Bi_M',
          'Sm_M', 'Eu_M', 'Gd_M', 'Tb_M', 'Dy_M', 'Ho_M', 'Er_M', 'Tm_M', 'Yb_M', 'Lu_M', 'Th_M', 'Pa_M', 'U_M']

K_TRANSITIONS = ['ka1', 'ka2', 'kb1', 'kb2']
L_TRANSITIONS = ['la1', 'la2', 'lb1', 'lb2', 'lb3', 'lb4', 'lb5',
          'lg1', 'lg2', 'lg3', 'lg4', 'll', 'ln']
M_TRANSITIONS = ['ma1', 'ma2', 'mb', 'mg']

TRANSITIONS_LOOKUP = {'K': K_TRANSITIONS, 'L': L_TRANSITIONS,
                      'M': M_TRANSITIONS}


def element_peak_xrf(x, area, center,
                     delta_center, delta_sigma,
                     ratio, ratio_adjust,
                     fwhm_offset, fwhm_fanoprime,
                     e_offset, e_linear, e_quadratic,
                     epsilon=2.96):
    """
    This is a function to construct xrf element peak, which is based on gauss profile,
    but more specific requirements need to be considered. For instance, the standard
    deviation is replaced by global fitting parameters, and energy calibration on x is
    taken into account.

    Parameters
    ----------
    x : array
        independent variable, channel number instead of energy
    area : float
        area of gaussian function
    center : float
        center position
    delta_center : float
        adjustment to center position
    delta_sigma : float
        adjustment to standard deviation
    ratio : float
        branching ratio
    ratio_adjust : float
        value used to adjust peak height
    fwhm_offset : float
        global fitting parameter for peak width
    fwhm_fanoprime : float
        global fitting parameter for peak width
    e_offset : float
        offset of energy calibration
    e_linear : float
        linear coefficient in energy calibration
    e_quadratic : float
        quadratic coefficient in energy calibration

    Returns
    -------
    array:
        gaussian peak profile
    """
    def get_sigma(center):
        temp_val = 2 * np.sqrt(2 * np.log(2))
        return np.sqrt((fwhm_offset/temp_val)**2 + center*epsilon*fwhm_fanoprime)

    x = e_offset + x * e_linear + x**2 * e_quadratic

    return gaussian(x, area, center+delta_center,
                    delta_sigma+get_sigma(center)) * ratio * ratio_adjust


class ElementModel(Model):

    __doc__ = _gen_class_docs(element_peak_xrf)

    def __init__(self, *args, **kwargs):
        super(ElementModel, self).__init__(element_peak_xrf, *args, **kwargs)
        self.set_param_hint('epsilon', value=2.96, vary=False)


def _set_parameter_hint(param_name, input_dict, input_model):
    """
    Set parameter hint information to lmfit model from input dict.

    .. warning :: This function mutates the input values.

    Parameters
    ----------
    param_name : str
        one of the fitting parameter name
    input_dict : dict
        all the initial values and constraints for given parameters
    input_model : object
        model object used in lmfit
    """
    if input_dict['bound_type'] == 'none':
        input_model.set_param_hint(name=param_name, value=input_dict['value'], vary=True)
    elif input_dict['bound_type'] == 'fixed':
        input_model.set_param_hint(name=param_name, value=input_dict['value'], vary=False)
    elif input_dict['bound_type'] == 'lohi':
        input_model.set_param_hint(name=param_name, value=input_dict['value'], vary=True,
                                   min=input_dict['min'], max=input_dict['max'])
    elif input_dict['bound_type'] == 'lo':
        input_model.set_param_hint(name=param_name, value=input_dict['value'], vary=True,
                                   min=input_dict['min'])
    elif input_dict['bound_type'] == 'hi':
        input_model.set_param_hint(name=param_name, value=input_dict['value'], vary=True,
                                   max=input_dict['max'])
    else:
        raise ValueError("could not set values for {0}".format(param_name))
    logger.debug(' {0} bound type: {1}, value: {2}, range: {3}'.
                 format(param_name, input_dict['bound_type'], input_dict['value'],
                            [input_dict['min'], input_dict['max']]))


def update_parameter_dict(xrf_parameter, fit_results):
    """
    Update fitting parameters dictionary according to given fitting results,
    usually obtained from previous run.

    .. warning :: This function mutates the input values.

    Parameters
    ----------
    xrf_parameter : dict
        saving all the fitting values and their bounds
    fit_results : object
        ModelFit object from lmfit
    """
    for k, v in six.iteritems(xrf_parameter):
        if k in fit_results.values:
            xrf_parameter[str(k)]['value'] = fit_results.values[str(k)]


_STRATEGY_REGISTRY = {'linear': sfb_pd.linear,
                      'adjust_element': sfb_pd.adjust_element,
                      'e_calibration': sfb_pd.e_calibration,
                      'fit_with_tail': sfb_pd.fit_with_tail,
                      'free_more': sfb_pd.free_more}


def register_strategy(key, strategy, overwrite=True):
    """
    Register new strategy.

    Parameters
    ----------
    key : str
        strategy name
    strategy : dict
        bound for every parameter
    """
    if (not overwrite) and (key in _STRATEGY_REGISTRY):
            if _STRATEGY_REGISTRY[key] is strategy:
                return
            raise RuntimeError("You are trying to overwrite an "
                               "existing strategy: {}".format(key))
    _STRATEGY_REGISTRY[key] = strategy


def set_parameter_bound(xrf_parameter, bound_option, extra_config=None):
    """
    Update the default value of bounds.

    .. warning :: This function mutates the input values.

    Parameters
    ----------
    xrf_parameter : dict
        saving all the fitting values and their bounds
    bound_option : str
        define bound type
    """
    strat_dict = _STRATEGY_REGISTRY[bound_option]
    if extra_config is None:
        extra_config = dict()
    for k, v in six.iteritems(xrf_parameter):
        if k == 'non_fitting_values':
            continue
        try:
            v['bound_type'] = strat_dict[k]
        except KeyError:
            v['bound_type'] = extra_config.get(k, 'fixed')


# This dict is used to update the current parameter dict to dynamically change
# the input data and do the fitting. The user can adjust parameters such as
# position, width, area or branching ratio.
PARAM_DEFAULTS = {'area': {'bound_type': 'none',
                           'max': 1000000000.0, 'min': 0, 'value': 1000},
                  'pos': {'bound_type': 'fixed',
                          'max': 0.005, 'min': -0.005, 'value': 0},
                  'ratio': {'bound_type': 'fixed',
                            'max': 5.0, 'min': 0.1, 'value': 1.0},
                  'width': {'bound_type': 'fixed',
                            'max': 0.02, 'min': -0.02, 'value': 0.0}}


class ParamController(object):
    """
    Update element peak information in parameter dictionary.
    This is an important step in dynamical fitting.
    """
    def __init__(self, params, elemental_lines):
        """
        Parameters
        ----------
        params : dict
            saving all the fitting values and their bounds
        elemental_lines : list
            e.g., ['Na_L', Mg_K', 'Pt_M'] refers to the L
            lines of Sodium, the K lines of Magnesium, and the M
            lines of Platinum
        """
        self._original_params = copy.deepcopy(params)
        self.params = copy.deepcopy(params)
        self.element_list = list(elemental_lines)  # to copy it
        self.element_linenames = get_activated_lines(
            self.params['coherent_sct_energy']['value'], self.element_list)
        self._element_strategy = dict()
        self._initialize_params()

    def _initialize_params(self):
        """
        Add all element information, such as pos, width, ratio into parameter dict.
        """
        for element in self.element_list:
            for kind in ['pos', 'width', 'area', 'ratio']:
                self.add_param(kind, element)

    def set_bound_type(self, bound_option):
        """
        Update the default value of bounds.

        Parameters
        ----------
        bound_option : str
            define bound type
        """
        set_parameter_bound(self.params, bound_option, self._element_strategy)

    def update_element_prop(self, element_list, **kwargs):
        """
        Update element properties, such as pos, width, area and ratio.

        Parameters
        ----------
        element_list : list
            define which element to update
        kws : dict
            define what kind of property to change

        Returns
        -------
        dict : updated value
        """
        for element in element_list:
            for kind, constraint in six.iteritems(kwargs):
                self.add_param(kind, element, constraint)

    def add_param(self, kind, element, constraint=None):
        """
        Create a Parameter controlling peak position, width,
        branching ratio, or area.

        Parameters
        ----------
        kind : {'pos', 'width', 'ratio', 'area'}
        element : str
            element name
        constraint : {'lo', 'hi', 'lohi', 'fixed', 'none'}, optional
            default "bound strategy" (fitting constraints)
        """
        if element not in self.element_list:
            self.element_list.append(element)
            self.element_linenames.extend(get_activated_lines(
                self.params['coherent_sct_energy']['value'], [element]))
        if kind == 'area':
            return self._set_area(element, constraint)
        element, line = element.split('_')
        transitions = LINES_LOOKUP[line]

        # Mg_L -> Mg_la1, which xraylib wants
        linenames = [
            '{element}_{transition}'.format(element, t) for t in transitions]

        PARAM_SUFFIXES = {'pos': '_delta_center',
                          'width': '_delta_sigma',
                          'ratio': '_ratio_adjust'}
        param_suffix = PARAM_SUFFIXES[kind]

        for linename in linenames:
            # check if the line is activated
            if linename not in self.element_linenames:
                continue
            param_name = str(linename) + '_delta_center'  # as in lmfit Model
            new_pos = PARAM_DEFAULTS[kind].copy()
            if constraint:
                self._element_strategy[param_name] = constraint 
            self.params.update({param_name: new_pos})

    def _add_area_param(self, element, constraint=None):
        """
        Special case for adding an area Parameter because
        we only ever fit the first peak area.

        Helper function called in self.add_param
        """
        if item in K_LINE:
            area_list = [str(item)+"_ka1_area"]
        elif item in L_LINE:
            item = item.split('_')[0]
            area_list = [str(item)+"_la1_area"]
        elif item in M_LINE:
            item = item.split('_')[0]
            area_list = [str(item)+"_ma1_area"]

        for linename in area_list:
            param_name = linename
            new_area = PARAM_DEFAULTS['area'].copy()
            if option:
                self._element_strategy[param_name] = option
            self.params.update({param_name: new_area})


def sum_area(element_name, result_val):
    """
    Return the total area for given element.

    Parameters
    ----------
    element_name : str
        name of given element
    result_val : obj
        result obj from lmfit to save all the fitting results

    Returns
    -------
    float
        the total area
    """
    def get_value(result_val, element_name, line_name):
        return (result_val.values[str(element_name)+'_'+line_name+'_area'] *
                result_val.values[str(element_name)+'_'+line_name+'_ratio'] *
                result_val.values[str(element_name)+'_'+line_name+'_ratio_adjust'])
    sumv = 0
    if element_name in K_LINE:
        for line_n in K_TRANSITIONS:
            full_name = element_name + '_' + line_n + '_area'
            if full_name in result_val.values:
                sumv += get_value(result_val, element_name, line_n)
    elif element_name in L_LINE:
        for line_n in L_TRANSITIONS:
            full_name = element_name.split('_')[0] + '_' + line_n + '_area'
            if full_name in result_val.values:
                sumv += get_value(result_val, element_name.split('_')[0], line_n)
    elif element_name in M_LINE:
        for line_n in M_TRANSITIONS:
            full_name = element_name.split('_')[0] + '_' + line_n + '_area'
            if full_name in result_val.values:
                sumv += get_value(result_val, element_name.split('_')[0], line_n)
    return sumv


class ModelSpectrum(object):
    """
    Construct Fluorescence spectrum which includes elastic peak,
    compton and element peaks.
    """

    def __init__(self, params, elements):
        """
        Parameters
        ----------
        params : dict
            saving all the fitting values and their bounds
        elements : list
        """
        self.params = copy.deepcopy(params)
        self.elements = list(elements)  # to copy
        self.incident_energy = self.params['coherent_sct_energy']['value']
        self.setup_compton_model()
        self.setup_elastic_model()

    def setup_compton_model(self):
        """
        setup parameters related to Compton model
        """
        compton = ComptonModel()

        compton_list = ['coherent_sct_energy', 'compton_amplitude',
                        'compton_angle', 'fwhm_offset', 'fwhm_fanoprime',
                        'e_offset', 'e_linear', 'e_quadratic',
                        'compton_gamma', 'compton_f_tail',
                        'compton_f_step', 'compton_fwhm_corr',
                        'compton_hi_gamma', 'compton_hi_f_tail']

        logger.debug(' ###### Started setting up parameters for compton model. ######')
        for name in compton_list:
            if name in self.params.keys():
                _set_parameter_hint(name, self.params[name], compton)
        logger.debug(' Finished setting up parameters for compton model.')

        self.compton_param = compton.make_params()
        self.compton = compton

    def setup_elastic_model(self):
        """
        setup parameters related to Elastic model
        """
        elastic = ElasticModel(prefix='elastic_')

        item = 'coherent_sct_amplitude'
        if item in self.params.keys():
            _set_parameter_hint(item, self.params[item], elastic)

        logger.debug(' ###### Started setting up parameters for elastic model. ######')

        # set constraints for the following global parameters
        elastic.set_param_hint('e_offset', value=self.compton_param['e_offset'].value,
                               expr='e_offset')
        elastic.set_param_hint('e_linear', value=self.compton_param['e_linear'].value,
                               expr='e_linear')
        elastic.set_param_hint('e_quadratic', value=self.compton_param['e_quadratic'].value,
                               expr='e_quadratic')
        elastic.set_param_hint('fwhm_offset', value=self.compton_param['fwhm_offset'].value,
                               expr='fwhm_offset')
        elastic.set_param_hint('fwhm_fanoprime', value=self.compton_param['fwhm_fanoprime'].value,
                               expr='fwhm_fanoprime')
        elastic.set_param_hint('coherent_sct_energy', value=self.compton_param['coherent_sct_energy'].value,
                               expr='coherent_sct_energy')
        logger.debug(' Finished setting up parameters for elastic model.')
        self.elastic = elastic

    def setup_element_model(self, element, default_area=1e5):
        """
        Construct element model.

        Parameters
        ----------
        element : str
            element name
        default_area : float, optional
            value for the initial area of a given element
            default is 1e5, found to be a good value
        """

        incident_energy = self.incident_energy
        parameter = self.params

        element_mod = None

        if element + '_K' in K_LINE:
            e = Element(element)
            if e.cs(incident_energy)['ka1'] == 0:
                logger.debug('%s Ka emission line is not activated '
                             'at this energy %f', element, incident_energy)
                return

            logger.debug(' --- Started building %s peak. ---', element)

            for num, item in enumerate(e.emission_line.all[:4]):
                line_name = item[0]
                val = item[1]

                if e.cs(incident_energy)[line_name] == 0:
                    continue

                gauss_mod = ElementModel(prefix=str(element)+'_'+str(line_name)+'_')
                gauss_mod.set_param_hint('e_offset', value=self.compton_param['e_offset'].value,
                                         expr='e_offset')
                gauss_mod.set_param_hint('e_linear', value=self.compton_param['e_linear'].value,
                                         expr='e_linear')
                gauss_mod.set_param_hint('e_quadratic', value=self.compton_param['e_quadratic'].value,
                                         expr='e_quadratic')
                gauss_mod.set_param_hint('fwhm_offset', value=self.compton_param['fwhm_offset'].value,
                                         expr='fwhm_offset')
                gauss_mod.set_param_hint('fwhm_fanoprime', value=self.compton_param['fwhm_fanoprime'].value,
                                         expr='fwhm_fanoprime')

                if line_name == 'ka1':
                    gauss_mod.set_param_hint('area', value=default_area, vary=True, min=0)
                    gauss_mod.set_param_hint('delta_center', value=0, vary=False)
                    gauss_mod.set_param_hint('delta_sigma', value=0, vary=False)
                elif line_name == 'ka2':
                    gauss_mod.set_param_hint('area', value=default_area, vary=True,
                                             expr=str(element)+'_ka1_'+'area')
                    gauss_mod.set_param_hint('delta_sigma', value=0, vary=False,
                                             expr=str(element)+'_ka1_'+'delta_sigma')
                    gauss_mod.set_param_hint('delta_center', value=0, vary=False,
                                             expr=str(element)+'_ka1_'+'delta_center')
                else:
                    gauss_mod.set_param_hint('area', value=default_area, vary=True,
                                             expr=str(element)+'_ka1_'+'area')
                    gauss_mod.set_param_hint('delta_center', value=0, vary=False)
                    gauss_mod.set_param_hint('delta_sigma', value=0, vary=False)

                area_name = str(element)+'_'+str(line_name)+'_area'
                if area_name in parameter:
                    _set_parameter_hint(area_name, parameter[area_name], gauss_mod)

                gauss_mod.set_param_hint('center', value=val, vary=False)
                ratio_v = e.cs(incident_energy)[line_name]/e.cs(incident_energy)['ka1']
                gauss_mod.set_param_hint('ratio', value=ratio_v, vary=False)
                gauss_mod.set_param_hint('ratio_adjust', value=1, vary=False)
                logger.debug(' {0} {1} peak is at energy {2} with'
                             ' branching ratio {3}.'. format(ename, line_name, val, ratio_v))

                # position needs to be adjusted
                pos_name = ename+'_'+str(line_name)+'_delta_center'
                if pos_name in parameter:
                    _set_parameter_hint('delta_center', parameter[pos_name],
                                        gauss_mod)

                # width needs to be adjusted
                width_name = ename+'_'+str(line_name)+'_delta_sigma'
                if width_name in parameter:
                    _set_parameter_hint('delta_sigma', parameter[width_name],
                                        gauss_mod)

                # branching ratio needs to be adjusted
                ratio_name = ename+'_'+str(line_name)+'_ratio_adjust'
                if ratio_name in parameter:
                    _set_parameter_hint('ratio_adjust', parameter[ratio_name],
                                        gauss_mod)

                if element_mod:
                    element_mod += gauss_mod
                else:
                    element_mod = gauss_mod
            logger.debug(' Finished building element peak for {0}'.format(ename))

        elif ename in L_LINE:
            ename = ename.split('_')[0]
            e = Element(ename)
            if e.cs(incident_energy)['la1'] == 0:
                logger.debug('{0} La1 emission line is not activated '
                             'at this energy {1}'.format(ename, incident_energy))
                return

            for num, item in enumerate(e.emission_line.all[4:-4]):

                line_name = item[0]
                val = item[1]

                if e.cs(incident_energy)[line_name] == 0:
                    continue

                gauss_mod = ElementModel(prefix=str(element)+'_'+str(line_name)+'_')

                gauss_mod.set_param_hint('e_offset', value=self.compton_param['e_offset'].value,
                                         expr='e_offset')
                gauss_mod.set_param_hint('e_linear', value=self.compton_param['e_linear'].value,
                                         expr='e_linear')
                gauss_mod.set_param_hint('e_quadratic', value=self.compton_param['e_quadratic'].value,
                                         expr='e_quadratic')
                gauss_mod.set_param_hint('fwhm_offset', value=self.compton_param['fwhm_offset'].value,
                                         expr='fwhm_offset')
                gauss_mod.set_param_hint('fwhm_fanoprime', value=self.compton_param['fwhm_fanoprime'].value,
                                         expr='fwhm_fanoprime')

                if line_name == 'la1':
                    gauss_mod.set_param_hint('area', value=default_area, vary=True)
                else:
                    gauss_mod.set_param_hint('area', value=default_area, vary=True,
                                             expr=str(element)+'_la1_'+'area')

                gauss_mod.set_param_hint('center', value=val, vary=False)
                gauss_mod.set_param_hint('sigma', value=1, vary=False)
                gauss_mod.set_param_hint('ratio',
                                         value=e.cs(incident_energy)[line_name]/e.cs(incident_energy)['la1'],
                                         vary=False)

                gauss_mod.set_param_hint('delta_center', value=0, vary=False)
                gauss_mod.set_param_hint('delta_sigma', value=0, vary=False)
                gauss_mod.set_param_hint('ratio_adjust', value=1, vary=False)

                # position needs to be adjusted
                pos_name = ename+'_'+str(line_name)+'_delta_center'
                if pos_name in parameter:
                    _set_parameter_hint('delta_center', parameter[pos_name],
                                        gauss_mod)

                # width needs to be adjusted
                width_name = ename+'_'+str(line_name)+'_delta_sigma'
                if width_name in parameter:
                    _set_parameter_hint('delta_sigma', parameter[width_name],
                                        gauss_mod)

                # branching ratio needs to be adjusted
                ratio_name = ename+'_'+str(line_name)+'_ratio_adjust'
                if ratio_name in parameter:
                    _set_parameter_hint('ratio_adjust', parameter[ratio_name],
                                        gauss_mod)
                if element_mod:
                    element_mod += gauss_mod
                else:
                    element_mod = gauss_mod

        elif ename in M_LINE:
            ename = ename.split('_')[0]
            e = Element(ename)
            if e.cs(incident_energy)['ma1'] == 0:
                logger.debug('{0} ma1 emission line is not activated '
                             'at this energy {1}'.format(ename, incident_energy))
                return

            for num, item in enumerate(e.emission_line.all[-4:]):

                line_name = item[0]
                val = item[1]

                if e.cs(incident_energy)[line_name] == 0:
                    continue

                gauss_mod = ElementModel(prefix=str(element)+'_'+str(line_name)+'_')

                gauss_mod.set_param_hint('e_offset', value=self.compton_param['e_offset'].value,
                                         expr='e_offset')
                gauss_mod.set_param_hint('e_linear', value=self.compton_param['e_linear'].value,
                                         expr='e_linear')
                gauss_mod.set_param_hint('e_quadratic', value=self.compton_param['e_quadratic'].value,
                                         expr='e_quadratic')
                gauss_mod.set_param_hint('fwhm_offset', value=self.compton_param['fwhm_offset'].value,
                                         expr='fwhm_offset')
                gauss_mod.set_param_hint('fwhm_fanoprime', value=self.compton_param['fwhm_fanoprime'].value,
                                         expr='fwhm_fanoprime')

                if line_name == 'ma1':
                    gauss_mod.set_param_hint('area', value=default_area, vary=True)
                else:
                    gauss_mod.set_param_hint('area', value=default_area, vary=True,
                                             expr=str(element)+'_ma1_'+'area')

                gauss_mod.set_param_hint('center', value=val, vary=False)
                gauss_mod.set_param_hint('sigma', value=1, vary=False)
                gauss_mod.set_param_hint('ratio',
                                         value=e.cs(incident_energy)[line_name]/e.cs(incident_energy)['ma1'],
                                         vary=False)

                gauss_mod.set_param_hint('delta_center', value=0, vary=False)
                gauss_mod.set_param_hint('delta_sigma', value=0, vary=False)
                gauss_mod.set_param_hint('ratio_adjust', value=1, vary=False)

                # position needs to be adjusted
                pos_name = ename+'_'+str(line_name)+'_delta_center'
                if pos_name in parameter:
                    _set_parameter_hint('delta_center', parameter[pos_name],
                                        gauss_mod)

                # width needs to be adjusted
                width_name = ename+'_'+str(line_name)+'_delta_sigma'
                if width_name in parameter:
                    _set_parameter_hint('delta_sigma', parameter[width_name],
                                        gauss_mod)

                # branching ratio needs to be adjusted
                ratio_name = ename+'_'+str(line_name)+'_ratio_adjust'
                if ratio_name in parameter:
                    _set_parameter_hint('ratio_adjust', parameter[ratio_name],
                                        gauss_mod)

                if element_mod:
                    element_mod += gauss_mod
                else:
                    element_mod = gauss_mod

        return element_mod

    def model_spectrum(self):
        """
        Put all models together to form a spectrum.
        """
        self.mod = self.compton + self.elastic

        for ename in self.element_list:
            self.mod += self.set_element_model(ename)

    def model_fit(self, x, y, w=None, method='leastsq', **kws):
        """
        Parameters
        ----------
        x : array
            independent variable
        y : array
            intensity
        w : array, optional
            weight for fitting
        method : str
            default as leastsq
        kws : dict
            fitting criteria, such as max number of iteration

        Returns
        -------
        result object from lmfit
        """

        pars = self.mod.make_params()
        result = self.mod.fit(y, pars, x=x, weights=w,
                              method=method, fit_kws=kws)
        return result


def trim(x, y, low, high):
    """
    Mask two arrays applying bounds to the first array.

    Parameters
    ----------
    x : array
        independent variable
    y : array
        dependent variable
    low : float
        low bound
    high : float
        high bound

    Returns
    -------
    array :
        x with new range
    array :
        y with new range
    """
    mask = (x > low) & (x < high)
    return x[mask], y[mask]


def get_escape_peak(y, ratio, fitting_parameters,
                    escape_e=1.73998):
    """
    Calculate the escape peak for given detector.

    Parameters
    ----------
    y : array
        original spectrum
    ratio : float
        ratio to adjust spectrum
    param : dict

    Returns
    -------
    array:
        x after shift, and adjusted y

    """
    x = np.arange(len(y))

    x = (fitting_parameters['e_offset']['value']
         + fitting_parameters['e_linear']['value']*x
         + fitting_parameters['e_quadratic']['value'] * x**2)

    return x-escape_e, y*ratio


def get_linear_model(x, param_dict, default_area=1e5):
    """
    Create spectrum with parameters given from param_dict.

    Parameters
    ----------
    x : array
        independent variable, channel number instead of energy
    param_dict : dict
        fitting paramters
    default_area : float
            value for the initial area of a given element

    Returns
    -------
    e_select : list
        selected elements for given energy
    matv : array
        matrix for linear fitting
    """
    MS = ModelSpectrum(param_dict)
    elist = MS.element_list

    e_select = []
    matv = []

    for i in range(len(elist)):
        e_model = MS.set_element_model(elist[i], default_area=default_area)
        if e_model:
            p = e_model.make_params()
            y_temp = e_model.eval(x=x, params=p)
            matv.append(y_temp)
            e_select.append(elist[i])

    p = MS.compton.make_params()
    y_temp = MS.compton.eval(x=x, params=p)
    matv.append(y_temp)

    p = MS.elastic.make_params()
    y_temp = MS.elastic.eval(x=x, params=p)
    matv.append(y_temp)

    matv = np.array(matv)
    matv = matv.transpose()
    return e_select, matv


class PreFitAnalysis(object):
    """
    It is used to automatic peak finding.
    """
    def __init__(self, experiments, standard):
        """
        Parameters
        ----------
        experiments : array
            spectrum of experiment data
        standard : array
            2D matrix of activated element spectrum
        """
        self.experiments = np.asarray(experiments)
        self.standard = np.asarray(standard)

    def nnls_fit(self):
        """
        Non-negative fitting.

        Returns
        -------
        results : array
            weights of different element
        residue : array
            error
        """
        standard = self.standard
        experiments = self.experiments

        [results, residue] = nnls(standard, experiments)
        return results, residue

    def nnls_fit_weight(self, c_weight=10):
        """
        Non-negative fitting with weight.

        Parameters
        ----------
        c_weight : float
            value used to calculate weight

        Returns
        -------
        results : array
            weights of different element
        residue : array
            error
        """
        standard = self.standard
        experiments = self.experiments

        weights = c_weight / (c_weight + experiments)
        weights = np.abs(weights)
        weights = weights/np.max(weights)

        a = np.transpose(np.multiply(np.transpose(standard), np.sqrt(weights)))
        b = np.multiply(experiments, np.sqrt(weights))

        [results, residue] = nnls(a, b)

        return results, residue


def pre_fit_linear(y0, param, element_list=K_LINE+L_LINE+M_LINE, weight=True):
    """
    Run prefit to get initial elements.

    Parameters
    ----------
    y0 : array
        Spectrum intensity
    param : dict
        Fitting parameters
    element_list : list, opt
        if not given, use list from param dict
    weight : bool
        use weight or not when fitting is performed

    Returns
    -------
    x : array
        x axis after cut
    result_dict : dict
        Fitting results
    """

    # Need to use deepcopy here to avoid unexpected change on parameter dict
    fitting_parameters = copy.deepcopy(param)

    x0 = np.arange(len(y0))

    # ratio to transfer energy value back to channel value
    approx_ratio = 100

    lowv = fitting_parameters['non_fitting_values']['energy_bound_low'] * approx_ratio
    highv = fitting_parameters['non_fitting_values']['energy_bound_high'] * approx_ratio

    x, y = trim(x0, y0, lowv, highv)

    if element_list:
        new_element = ', '.join(element_list)
        fitting_parameters['non_fitting_values']['element_list'] = new_element

    e_select, matv = get_linear_model(x, fitting_parameters)

    non_element = ['compton', 'elastic']
    total_list = e_select + non_element
    total_list = [str(v) for v in total_list]

    # get background
    bg = snip_method(y, fitting_parameters['e_offset']['value'],
                     fitting_parameters['e_linear']['value'],
                     fitting_parameters['e_quadratic']['value'])
    y = y - bg

    PF = PreFitAnalysis(y, matv)

    if weight:
        out, res = PF.nnls_fit_weight()
    else:
        out, res = PF.nnls_fit()

    total_y = out * matv

    # use ordered dict
    result_dict = OrderedDict()

    for i in range(len(total_list)):
        if np.sum(total_y[:, i]) == 0:
            continue
        if '_L' in total_list[i] or '_M' in total_list[i] \
                or total_list[i] in non_element:
            result_dict.update({total_list[i]: total_y[:, i]})
        else:
            result_dict.update({total_list[i] + '_K': total_y[:, i]})
    result_dict.update(background=bg)

    x = (fitting_parameters['e_offset']['value'] +
         fitting_parameters['e_linear']['value']*x +
         fitting_parameters['e_quadratic']['value'] * x**2)

    return x, result_dict


def get_activated_lines(incident_energy, element_names):
    """
    Parameters
    ----------
    incident_energy : float
        beam energy
    element_names : list
        e.g., ['Na_K', 'Mg_L', 'Pt_M']

    Returns
    -------
    list
        all activated lines for given elements
    """
    lines = []
    for v in element_names:
        activated_line = _get_activated_line(incident_energy, v)
        if activated_line:
            lines.extend(activated_line)
    return lines


def _get_activated_line(incident_energy, ename):
    """
    Collect all the activated lines for given element.

    Parameters
    ----------
    incident_energy : float
        beam energy
    ename : str
        element name

    Returns
    -------
    list
        all possible line names for given element
    """
    line_list = []
    if element in K_LINE:
        e = Element(element)
        if e.cs(incident_energy)['ka1'] == 0:
            return
        for num, item in enumerate(e.emission_line.all[:4]):
            line_name = item[0]
            if e.cs(incident_energy)[line_name] == 0:
                continue
            line_list.append(str(element)+'_'+str(line_name))
        return line_list

    elif element in L_LINE:
        element = element.split('_')[0]
        e = Element(element)
        if e.cs(incident_energy)['la1'] == 0:
            return
        for num, item in enumerate(e.emission_line.all[4:-4]):
            line_name = item[0]
            if e.cs(incident_energy)[line_name] == 0:
                continue
            line_list.append(str(element)+'_'+str(line_name))
        return line_list

    elif element in M_LINE:
        element = element.split('_')[0]
        e = Element(element)
        if e.cs(incident_energy)['ma1'] == 0:
            return
        for num, item in enumerate(e.emission_line.all[-4:]):
            line_name = item[0]
            if e.cs(incident_energy)[line_name] == 0:
                continue
            line_list.append(str(element)+'_'+str(line_name))
        return line_list
