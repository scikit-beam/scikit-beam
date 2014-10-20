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
import six

import logging
logger = logging.getLogger(__name__)

from nsls2.constants import Element
from nsls2.fitting.model.physics_peak import (gauss_peak)
from nsls2.fitting.model.physics_model import (ComptonModel, ElasticModel,
                                               _gen_class_docs)
from nsls2.fitting.base.parameter_data import get_para
from nsls2.fitting.model.background import snip_method
from lmfit import Model


k_line = ['Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
          'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
          'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
          'In', 'Sn', 'Sb', 'Te', 'I', 'dummy', 'dummy']

l_line = ['Mo_L', 'Tc_L', 'Ru_L', 'Rh_L', 'Pd_L', 'Ag_L', 'Cd_L', 'In_L', 'Sn_L', 'Sb_L', 'Te_L', 'I_L', 'Xe_L', 'Cs_L', 'Ba_L', 'La_L', 'Ce_L', 'Pr_L', 'Nd_L', 'Pm_L', 'Sm_L',
          'Eu_L', 'Gd_L', 'Tb_L', 'Dy_L', 'Ho_L', 'Er_L', 'Tm_L', 'Yb_L', 'Lu_L', 'Hf_L', 'Ta_L', 'W_L', 'Re_L', 'Os_L', 'Ir_L', 'Pt_L', 'Au_L', 'Hg_L', 'Tl_L',
          'Pb_L', 'Bi_L', 'Po_L', 'At_L', 'Rn_L', 'Fr_L', 'Ac_L', 'Th_L', 'Pa_L', 'U_L', 'Np_L', 'Pu_L', 'Am_L', 'Br_L', 'Ga_L']

m_line = ['Au_M', 'Pb_M', 'U_M', 'noise', 'Pt_M', 'Ti_M', 'Gd_M', 'dummy', 'dummy']


def gauss_peak_xrf(x, area, center,
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
        independent variable
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

    return gauss_peak(x, area, center+delta_center,
                      delta_sigma+get_sigma(center)) * ratio * ratio_adjust


class GaussModel_xrf(Model):

    __doc__ = _gen_class_docs(gauss_peak_xrf)

    def __init__(self, *args, **kwargs):
        super(GaussModel_xrf, self).__init__(gauss_peak_xrf, *args, **kwargs)
        self.set_param_hint('epsilon', value=2.96, vary=False)


def _set_parameter_hint(para_name, input_dict, input_model,
                        log_option=False):
    """
    Set parameter information to a given model

    Parameters
    ----------
    para_name : str
        parameter used for fitting
    input_dict : dict
        all the initial values and constraints for given parameters
    input_model : object
        model object used in lmfit
    log_option : bool
        option for logger
    """

    if input_dict['bound_type'] == 'none':
        input_model.set_param_hint(name=para_name, value=input_dict['value'], vary=True)
    elif input_dict['bound_type'] == 'fixed':
        input_model.set_param_hint(name=para_name, value=input_dict['value'], vary=False)
    elif input_dict['bound_type'] == 'lohi':
        input_model.set_param_hint(name=para_name, value=input_dict['value'], vary=True,
                                   min=input_dict['min'], max=input_dict['max'])
    elif input_dict['bound_type'] == 'lo':
        input_model.set_param_hint(name=para_name, value=input_dict['value'], vary=True,
                                   min=input_dict['min'])
    elif input_dict['bound_type'] == 'hi':
        input_model.set_param_hint(name=para_name, value=input_dict['value'], vary=True,
                                   min=input_dict['max'])
    else:
        raise ValueError("could not set values for {0}".format(para_name))
    if log_option:
        logger.info(' {0} bound type: {1}, value: {2}, range: {3}'.
                    format(para_name, input_dict['bound_type'], input_dict['value'],
                           [input_dict['min'], input_dict['max']]))
    return


def update_parameter_dict(xrf_parameter, fit_results):
    """
    Update fitting parameters dictionary according to given fitting results,
    usually obtained from previous run.

    Parameters
    ----------
    xrf_parameter : dict
        saving all the fitting values and their bounds
    fit_results : object
        ModelFit object from lmfit
    """
    for k, v in six.iteritems(xrf_parameter):
        if fit_results.values.has_key(k):
            xrf_parameter[str(k)]['value'] = fit_results.values[str(k)]


def set_parameter_bound(xrf_parameter, bound_option):
    """
    Update the default value of bounds.

    Parameters
    ----------
    xrf_parameter : dict
        saving all the fitting values and their bounds
    bound_option : str
        define bound type
    """
    for k, v in six.iteritems(xrf_parameter):
        if k == 'non_fitting_values':
            continue
        v['bound_type'] = v[str(bound_option)]

    return


element_dict = {
    'pos': {"bound_type": "fixed", "min": -0.005, "max": 0.005, "value": 0,
            "free_more": "fixed", "adjust_element": "lohi", "e_calibration": "fixed", "linear": "fixed"},
    'width': {"bound_type": "fixed", "min": -0.02, "max": 0.02, "value": 0.0,
              "free_more": "fixed", "adjust_element": "lohi", "e_calibration": "fixed", "linear": "fixed"},
    'area': {"bound_type": "none", "min": 0, "max": 1e9, "value": 1000,
             "free_more": "none", "adjust_element": "fixed", "e_calibration": "fixed", "linear": "none"},
    'ratio': {"bound_type": "fixed", "min": 0.1, "max": 5.0, "value": 1.0,
              "free_more": "fixed", "adjust_element": "lohi", "e_calibration":"fixed", "linear":"fixed"}
}


def get_L_line(prop_name, element):
    #l_list = ['la1', 'la2', 'lb1', 'lb2', 'lb3', 'lb4', 'lb5',
    #          'lg1', 'lg2', 'lg3', 'lg4', 'll', 'ln']
    l_list = ['la1', 'la2', 'lb1', 'lb2', 'lb3',
              'lg1', 'lg2', 'll']
    return [str(prop_name)+'-'+str(element)+'-'+str(item)
            for item in l_list]


class ElementController(object):

    def __init__(self, xrf_parameter, fit_name):
        """
        Update element peak information in parameter dictionary.

        Parameters
        ----------
        xrf_parameter : dict
            saving all the fitting values and their bounds

        """
        self.new_parameter = xrf_parameter.copy()
        self.element_name = [item[0:-5] for item in fit_name if 'area' in item]

    def set_val(self, element_list,
                **kws):
        """
        element_list : list
            define which element to update
        kws : dict
            define what kind of property to change
        """

        for k, v in six.iteritems(kws):
            if k == 'pos':
                func = self.set_position
            elif k == 'width':
                func = self.set_width
            elif k == 'area':
                func = self.set_area
            elif k == 'ratio':
                func = self.set_ratio
            else:
                raise ValueError('Please define either pos, width or area.')

            for element in element_list:
                func(element, option=v)

        return self.new_parameter

    def set_position(self, item, option=None):
        """
        Parameters
        ----------
        item : str
            element name
        option : str, optional
            way to control position
        """

        if item in k_line:
            pos_list = [str(item)+"_ka1_delta_center",
                        str(item)+"_ka2_delta_center",
                        str(item)+"_kb1_delta_center"]
            for linename in pos_list:
                new_pos = element_dict['pos'].copy()
                if option:
                    new_pos['adjust_element'] = option
                addv = {linename: new_pos}
                self.new_parameter.update(addv)

        elif item in l_line:
            item = item[0:-2]
            pos_list = get_L_line('pos', item)
            for linename in pos_list:
                linev = linename.split('-')[1]+'_'+linename.split('-')[2]
                if linev not in self.element_name:
                    continue
                new_pos = element_dict['pos'].copy()
                if option:
                    new_pos['adjust_element'] = option
                addv = {linename: new_pos}
                self.new_parameter.update(addv)

        return

    def set_width(self, item, option=None):
        """
        Parameters
        ----------
        item : str
            element name
        option : str, optional
            way to control width
        """
        if item in k_line:
            width_list = [str(item)+"_ka1_delta_sigma",
                          str(item)+"_ka2_delta_sigma",
                          str(item)+"_kb1_delta_sigma"]
            for linename in width_list:
                new_width = element_dict['width'].copy()
                if option:
                    new_width['adjust_element'] = option
                addv = {linename: new_width}
                self.new_parameter.update(addv)
        elif item in l_line:
            item = item[0:-2]
            data_list = get_L_line('width', item)
            for linename in data_list:
                linev = linename.split('-')[1]+'_'+linename.split('-')[2]
                if linev not in self.element_name:
                    continue
                new_val = element_dict['width'].copy()
                if option:
                    new_val['adjust_element'] = option
                addv = {linename: new_val}
                self.new_parameter.update(addv)
        return

    def set_area(self, item, option=None):
        """
        Parameters
        ----------
        item : str
            element name
        option : str, optional
            way to control area
        """
        if item in k_line:
            area_list = [str(item)+"_ka1_area"]
            for linename in area_list:
                new_area = element_dict['area'].copy()
                if option:
                    new_area['adjust_element'] = option
                addv = {linename: new_area}
                self.new_parameter.update(addv)
        elif item in l_line:
            item = item[0:-2]
            data_list = get_L_line('area', item)
            for linename in data_list:
                linev = linename.split('-')[1]+'_'+linename.split('-')[2]
                if linev not in self.element_name:
                    continue
                new_val = element_dict['area'].copy()
                if option:
                    new_val['adjust_element'] = option
                addv = {linename: new_val}
                self.new_parameter.update(addv)
        return

    def set_ratio(self, item, option=None):
        """
        Parameters
        ----------
        item : str
            element name
        option : str, optional
            way to control branching ratio
        """
        if item in k_line:
            data_list = [str(item)+"_kb1_ratio_adjust"]
            for linename in data_list:
                new_val = element_dict['ratio'].copy()
                if option:
                    new_val['adjust_element'] = option
                addv = {linename: new_val}
                self.new_parameter.update(addv)

        elif item in l_line:
            item = item[0:-2]
            data_list = get_L_line('ratio', item)
            for linename in data_list:
                if 'la1' in linename:
                    continue
                linev = linename.split('-')[1]+'_'+linename.split('-')[2]
                if linev not in self.element_name:
                    continue
                new_val = element_dict['ratio'].copy()
                if option:
                    new_val['adjust_element'] = option
                addv = {linename: new_val}
                self.new_parameter.update(addv)
        return


def get_sum_area(element_name, result_val):
    if element_name in k_line:
        sum = result_val.values[str(element_name)+'_ka1_area'] + \
              result_val.values[str(element_name)+'_ka2_area'] + \
              result_val.values[str(element_name)+'_kb1_area']
        return sum


class ModelSpectrum(object):

    def __init__(self, xrf_parameter):
        """
        Parameters
        ----------
        xrf_parameter : dict
            saving all the fitting values and their bounds
        """

        self.parameter = xrf_parameter

        non_fit = self.parameter['non_fitting_values']
        if non_fit.has_key('element_list'):
            if ',' in non_fit['element_list']:
                self.element_list = non_fit['element_list'].split(', ')
            else:
                self.element_list = non_fit['element_list'].split()
            self.element_list = [item.strip() for item in self.element_list]
        else:
            logger.critical(' No element is selected for fitting!')

        self.incident_energy = self.parameter['coherent_sct_energy']['value']

        self.parameter_default = get_para()

        self.model_spectrum()

        return

    def setComptonModel(self):
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
            if name in self.parameter.keys():
                _set_parameter_hint(name, self.parameter[name], compton)
            else:
                _set_parameter_hint(name, self.parameter_default[name], compton)
        logger.debug(' Finished setting up parameters for compton model.')
        return compton

    def setElasticModel(self):
        """
        setup parameters related to Elastic model
        """
        elastic = ElasticModel(prefix='elastic_')

        item = 'coherent_sct_amplitude'
        if item in self.parameter.keys():
            _set_parameter_hint(item, self.parameter[item], elastic)
        else:
            _set_parameter_hint(item, self.parameter_default[item], elastic)

        logger.debug(' ###### Started setting up parameters for elastic model. ######')

        # set constraints for the following global parameters
        elastic.set_param_hint('e_offset', expr='e_offset')
        elastic.set_param_hint('e_linear', expr='e_linear')
        elastic.set_param_hint('e_quadratic', expr='e_quadratic')
        elastic.set_param_hint('fwhm_offset', expr='fwhm_offset')
        elastic.set_param_hint('fwhm_fanoprime', expr='fwhm_fanoprime')
        elastic.set_param_hint('coherent_sct_energy', expr='coherent_sct_energy')
        logger.debug(' Finished setting up parameters for elastic model.')

        return elastic

    def model_spectrum(self):
        """
        Add all element peaks to the model.
        """
        incident_energy = self.incident_energy
        element_list = self.element_list
        parameter = self.parameter

        mod = self.setComptonModel() + self.setElasticModel()

        for ename in element_list:
            if ename in k_line:
                e = Element(ename)
                if e.cs(incident_energy)['ka1'] == 0:
                    logger.info(' {0} Ka emission line is not activated '
                                'at this energy {1}'.format(ename, incident_energy))
                    continue

                logger.debug(' --- Started building {0} peak. ---'.format(ename))

                for num, item in enumerate(e.emission_line.all[:4]):
                    line_name = item[0]
                    val = item[1]

                    if e.cs(incident_energy)[line_name] == 0:
                        continue

                    gauss_mod = GaussModel_xrf(prefix=str(ename)+'_'+str(line_name)+'_')
                    gauss_mod.set_param_hint('e_offset', expr='e_offset')
                    gauss_mod.set_param_hint('e_linear', expr='e_linear')
                    gauss_mod.set_param_hint('e_quadratic', expr='e_quadratic')
                    gauss_mod.set_param_hint('fwhm_offset', expr='fwhm_offset')
                    gauss_mod.set_param_hint('fwhm_fanoprime', expr='fwhm_fanoprime')

                    if line_name == 'ka1':
                        gauss_mod.set_param_hint('area', value=100, vary=True, min=0)
                        gauss_mod.set_param_hint('delta_center', value=0, vary=False)
                        gauss_mod.set_param_hint('delta_sigma', value=0, vary=False)
                    elif line_name == 'ka2':
                        gauss_mod.set_param_hint('area', value=100, vary=True,
                                                 expr=str(ename)+'_ka1_'+'area')
                        gauss_mod.set_param_hint('delta_sigma', value=0, vary=False,
                                                 expr=str(ename)+'_ka1_'+'delta_sigma')
                        gauss_mod.set_param_hint('delta_center', value=0, vary=False,
                                                 expr=str(ename)+'_ka1_'+'delta_center')
                    else:
                        gauss_mod.set_param_hint('area', value=100, vary=True,
                                                 expr=str(ename)+'_ka1_'+'area')
                        gauss_mod.set_param_hint('delta_center', value=0, vary=False)
                        gauss_mod.set_param_hint('delta_sigma', value=0, vary=False)

                    #gauss_mod.set_param_hint('delta_center', value=0, vary=False)
                    #gauss_mod.set_param_hint('delta_sigma', value=0, vary=False)

                    area_name = str(ename)+'_'+str(line_name)+'_area'
                    if parameter.has_key(area_name):
                        _set_parameter_hint(area_name, parameter[area_name],
                                            gauss_mod, log_option=True)

                    gauss_mod.set_param_hint('center', value=val, vary=False)
                    ratio_v = e.cs(incident_energy)[line_name]/e.cs(incident_energy)['ka1']
                    gauss_mod.set_param_hint('ratio', value=ratio_v, vary=False)
                    gauss_mod.set_param_hint('ratio_adjust', value=1, vary=False)
                    logger.info(' {0} {1} peak is at energy {2} with'
                                ' branching ratio {3}.'. format(ename, line_name, val, ratio_v))

                    # position needs to be adjusted
                    pos_name = ename+'_'+str(line_name)+'_delta_center'
                    if parameter.has_key(pos_name):
                        _set_parameter_hint('delta_center', parameter[pos_name],
                                            gauss_mod, log_option=True)

                    # width needs to be adjusted
                    width_name = ename+'_'+str(line_name)+'_delta_sigma'
                    if parameter.has_key(width_name):
                        _set_parameter_hint('delta_sigma', parameter[width_name],
                                            gauss_mod, log_option=True)

                    # branching ratio needs to be adjusted
                    ratio_name = ename+'_'+str(line_name)+'_ratio_adjust'
                    if parameter.has_key(ratio_name):
                        _set_parameter_hint('ratio_adjust', parameter[ratio_name],
                                            gauss_mod, log_option=True)

                    mod = mod + gauss_mod
                logger.debug(' Finished building element peak for {0}'.format(ename))

            elif ename in l_line:
                ename = ename[:-2]
                e = Element(ename)
                if e.cs(incident_energy)['la1'] == 0:
                    logger.info('{0} La1 emission line is not activated '
                                'at this energy {1}'.format(ename, incident_energy))
                    continue

                for num, item in enumerate(e.emission_line.all[4:-4]):

                    line_name = item[0]
                    val = item[1]

                    if e.cs(incident_energy)[line_name] == 0:
                        continue

                    gauss_mod = GaussModel_xrf(prefix=str(ename)+'_'+str(line_name)+'_')

                    gauss_mod.set_param_hint('e_offset', expr='e_offset')
                    gauss_mod.set_param_hint('e_linear', expr='e_linear')
                    gauss_mod.set_param_hint('e_quadratic', expr='e_quadratic')
                    gauss_mod.set_param_hint('fwhm_offset', expr='fwhm_offset')
                    gauss_mod.set_param_hint('fwhm_fanoprime', expr='fwhm_fanoprime')

                    if line_name == 'la1':
                        gauss_mod.set_param_hint('area', value=100, vary=True)
                                             #expr=gauss_mod.prefix+'ratio_val * '+str(ename)+'_la1_'+'area')
                    else:
                        gauss_mod.set_param_hint('area', value=100, vary=True,
                                                 expr=str(ename)+'_la1_'+'area')

                    gauss_mod.set_param_hint('center', value=val, vary=False)
                    gauss_mod.set_param_hint('sigma', value=1, vary=False)
                    gauss_mod.set_param_hint('ratio',
                                             value=e.cs(incident_energy)[line_name]/e.cs(incident_energy)['la1'],
                                             vary=False)

                    gauss_mod.set_param_hint('delta_center', value=0, vary=False)
                    gauss_mod.set_param_hint('delta_sigma', value=0, vary=False)
                    gauss_mod.set_param_hint('ratio_adjust', value=1, vary=False)

                    # position needs to be adjusted
                    if ename in pos_adjust:
                        pos_name = 'pos-'+ename+'-'+str(line_name)
                        if parameter.has_key(pos_name):
                            _set_parameter_hint('delta_center', parameter[pos_name],
                                                gauss_mod, log_option=True)

                    # width needs to be adjusted
                    if ename in width_adjust:
                        width_name = 'width-'+ename+'-'+str(line_name)
                        if parameter.has_key(width_name):
                            _set_parameter_hint('delta_sigma', parameter[width_name],
                                                gauss_mod, log_option=True)


                    # branching ratio needs to be adjusted
                    if ename in ratio_adjust:
                        ratio_name = 'ratio-'+ename+'-'+str(line_name)
                        if parameter.has_key(ratio_name):
                            _set_parameter_hint('ratio_adjust', parameter[ratio_name],
                                                gauss_mod, log_option=True)

                    mod = mod + gauss_mod

            elif ename in m_line:
                ename = ename[:-2]
                e = Element(ename)
                #if e.cs(incident_energy)['ma1'] == 0:
                #    logger.info('{0} ma1 emission line is not activated '
                #                'at this energy {1}'.format(ename, incident_energy))
                #    continue

                for num, item in enumerate(e.emission_line.all[-4:]):

                    line_name = item[0]
                    val = item[1]

                    #if e.cs(incident_energy)[line_name] == 0:
                    #    continue

                    gauss_mod = GaussModel_xrf(prefix=str(ename)+'_'+str(line_name)+'_')

                    gauss_mod.set_param_hint('e_offset', expr='e_offset')
                    gauss_mod.set_param_hint('e_linear', expr='e_linear')
                    gauss_mod.set_param_hint('e_quadratic', expr='e_quadratic')
                    gauss_mod.set_param_hint('fwhm_offset', expr='fwhm_offset')
                    gauss_mod.set_param_hint('fwhm_fanoprime', expr='fwhm_fanoprime')

                    if line_name == 'ma1':
                        gauss_mod.set_param_hint('area', value=100, vary=True)
                    else:
                        gauss_mod.set_param_hint('area', value=100, vary=True,
                                                 expr=str(ename)+'_ma1_'+'area')

                    gauss_mod.set_param_hint('center', value=val, vary=False)
                    gauss_mod.set_param_hint('sigma', value=1, vary=False)
                    gauss_mod.set_param_hint('ratio',
                                             value=0.1, #e.cs(incident_energy)[line_name]/e.cs(incident_energy)['ma1'],
                                             vary=False)

                    gauss_mod.set_param_hint('delta_center', value=0, vary=False)
                    gauss_mod.set_param_hint('delta_sigma', value=0, vary=False)
                    gauss_mod.set_param_hint('ratio_adjust', value=1, vary=False)

                    mod = mod + gauss_mod


        self.mod = mod
        return

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
        obj
            saving all the fitting results
        """

        pars = self.mod.make_params()
        result = self.mod.fit(y, pars, x=x, weights=w,
                              method=method, fit_kws=kws)
        return result


def set_range(parameter, x1, y1):

    lowv = parameter['non_fitting_values']['energy_bound_low'] * 100
    highv = parameter['non_fitting_values']['energy_bound_high'] * 100

    all = zip(x1, y1)

    x0 = [item[0] for item in all if item[0] > lowv and item[0] < highv]
    y0 = [item[1] for item in all if item[0] > lowv and item[0] < highv]
    return np.array(x0), np.array(y0)


def construct_linear_model(x, energy, element_list):
    """
    Construct linear model for fluorescence analysis.

    """
    pass



