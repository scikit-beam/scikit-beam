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
import json
import os

import logging
logger = logging.getLogger(__name__)

from nsls2.constants import Element
from nsls2.fitting.model.physics_peak import (gauss_peak)
from nsls2.fitting.model.physics_model import (ComptonModel, ElasticModel,
                                               _gen_class_docs)
from nsls2.fitting.base.parameter_data import get_para
from lmfit import Model


k_line = ['Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
          'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
          'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
          'In', 'Sn', 'Sb', 'Te', 'I', 'dummy', 'dummy']

l_line = ['Mo_L', 'Tc_L', 'Ru_L', 'Rh_L', 'Pd_L', 'Ag_L', 'Cd_L', 'In_L', 'Sn_L', 'Sb_L', 'Te_L', 'I_L', 'Xe_L', 'Cs_L', 'Ba_L', 'La_L', 'Ce_L', 'Pr_L', 'Nd_L', 'Pm_L', 'Sm_L',
          'Eu_L', 'Gd_L', 'Tb_L', 'Dy_L', 'Ho_L', 'Er_L', 'Tm_L', 'Yb_L', 'Lu_L', 'Hf_L', 'Ta_L', 'W_L', 'Re_L', 'Os_L', 'Ir_L', 'Pt_L', 'Au_L', 'Hg_L', 'Tl_L',
          'Pb_L', 'Bi_L', 'Po_L', 'At_L', 'Rn_L', 'Fr_L', 'Ac_L', 'Th_L', 'Pa_L', 'U_L', 'Np_L', 'Pu_L', 'Am_L', 'Br_L', 'Ga_L']

m_line = ['Au_M', 'Pb_M', 'U_M', 'noise', 'Pt_M', 'Ti_M', 'Gd_M', 'dummy', 'dummy']


def gauss_peak_xrf(x, area, center, sigma,
                   ratio, fwhm_offset, fwhm_fanoprime,
                   epsilon=2.96):
    """
    This is a function to construct xrf element peak, which is based on gauss profile,
    but more specific requirements need to be considered.

    Parameters
    ----------
    x : array
        independent variable
    area : float
        area of gaussian function
    center : float
        center position
    sigma : float
        standard deviation
    ratio : float
        value used to adjust peak height
    fwhm_offset : float
        global fitting parameter for peak width
    fwhm_fanoprime : float
        global fitting parameter for peak width

    Returns
    -------
    array:
        gaussian peak profile
    """
    def get_sigma(center):
        temp_val = 2 * np.sqrt(2 * np.log(2))
        return np.sqrt((fwhm_offset/temp_val)**2 + center*epsilon*fwhm_fanoprime)

    return gauss_peak(x, area, center, sigma*get_sigma(center)) * ratio


class GaussModel_xrf(Model):

    __doc__ = _gen_class_docs(gauss_peak_xrf)

    def __init__(self, *args, **kwargs):
        super(GaussModel_xrf, self).__init__(gauss_peak_xrf, *args, **kwargs)
        self.set_param_hint('epsilon', value=2.96, vary=False)


def _set_value(para_name, input_dict, model_name):
    """
    Set parameter information to a given model

    Parameters
    ----------
    para_name : str
        parameter used for fitting
    input_dict : dict
        all the initial values and constraints for given parameters
    model_name : object
        model object used in lmfit
    """

    if input_dict['bound_type'] == 'none':
        model_name.set_param_hint(name=para_name, value=input_dict['value'], vary=True)
    elif input_dict['bound_type'] == 'fixed':
        model_name.set_param_hint(name=para_name, value=input_dict['value'], vary=False)
    elif input_dict['bound_type'] == 'lohi':
        model_name.set_param_hint(name=para_name, value=input_dict['value'], vary=True,
                                  min=input_dict['min'], max=input_dict['max'])
    elif input_dict['bound_type'] == 'lo':
        model_name.set_param_hint(name=para_name, value=input_dict['value'], vary=True,
                                  min=input_dict['min'])
    elif input_dict['bound_type'] == 'hi':
        model_name.set_param_hint(name=para_name, value=input_dict['value'], vary=True,
                                  min=input_dict['max'])
    else:
        raise ValueError("could not set values for {0}".format(para_name))
    return


class ModelSpectrum(object):

    def __init__(self, config_file='xrf_paramter.json'):
        """
        Parameters
        ----------
        config_file : str
            file save all the fitting parameters
        """
        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 config_file)

        if not os.path.exists(file_path):
            logger.critical('No configuration file can be found.')
        else:
            with open(file_path, 'r') as json_data:
                self.parameter = json.load(json_data)
                logger.info('Read data from configuration file {0}.'.format(file_path))

        if ',' in self.parameter['element_list']:
            self.element_list = self.parameter['element_list'].split(', ')
        else:
            self.element_list = self.parameter['element_list'].split()
        self.element_list = [item.strip() for item in self.element_list]

        self.incident_energy = self.parameter['coherent_sct_energy']['value']

        self.parameter_default = get_para()
        return

    def setComptonModel(self):
        """
        setup parameters related to Compton model
        """
        compton = ComptonModel()

        compton_list = ['coherent_sct_energy', 'compton_amplitude',
                        'compton_angle', 'fwhm_offset', 'fwhm_fanoprime',
                        'compton_gamma', 'compton_f_tail',
                        'compton_f_step', 'compton_fwhm_corr',
                        'compton_hi_gamma', 'compton_hi_f_tail']

        logger.debug('Set up parameters for compton model')
        for name in compton_list:
            if name in self.parameter.keys():
                _set_value(name, self.parameter[name], compton)
            else:
                _set_value(name, self.parameter_default[name], compton)
        logger.debug('Finish setting up paramters for compton model')
        return compton

    def setElasticModel(self):
        """
        setup parameters related to Elastic model
        """
        elastic = ElasticModel(prefix='e_')

        item = 'coherent_sct_amplitude'
        if item in self.parameter.keys():
            _set_value(item, self.parameter[item], elastic)
        else:
            _set_value(item, self.parameter_default[item], elastic)

        logger.debug('Set up paramters for elastic model')

        # set constraints for the following global parameters
        elastic.set_param_hint(name='fwhm_offset', value=0.1, vary=True, expr='fwhm_offset')
        elastic.set_param_hint(name='fwhm_fanoprime', value=0.0, vary=True, expr='fwhm_fanoprime')
        elastic.set_param_hint(name='coherent_sct_energy', value=self.incident_energy,
                               expr='coherent_sct_energy')
        logger.debug('Finish setting up paramters for elastic model')

        return elastic

    def model_spectrum(self):
        """
        Add all element peaks to the model.
        """
        incident_energy = self.incident_energy
        element_list = self.element_list
        parameter = self.parameter

        mod = self.setComptonModel() + self.setElasticModel()

        element_adjust = []
        if parameter.has_key('element'):
            element_adjust = parameter['element'].keys()
            logger.info('Those elements need to be adjusted {0}'.format(element_adjust))
        else:
            logger.info('No adjustment needs to be considered '
                        'on the position and width of element peak.')

        for ename in element_list:
            if ename in k_line:
                e = Element(ename)
                if e.cs(incident_energy)['ka1'] == 0:
                    logger.info('{0} Ka emission line is not activated '
                                'at this energy {1}'.format(ename, incident_energy))
                    continue

                logger.debug('Started building element peak for {0}'.format(ename))

                for num, item in enumerate(e.emission_line.all[:4]):
                    line_name = item[0]
                    val = item[1]

                    if e.cs(incident_energy)[line_name] == 0:
                        continue

                    gauss_mod = GaussModel_xrf(prefix=str(ename)+'_'+str(line_name)+'_')
                    gauss_mod.set_param_hint('fwhm_offset', value=0.1, vary=True, expr='fwhm_offset')
                    gauss_mod.set_param_hint('fwhm_fanoprime', value=0.1, vary=True, expr='fwhm_fanoprime')

                    if line_name == 'ka1':
                        gauss_mod.set_param_hint('area', value=100, vary=True, min=0)
                    else:
                        gauss_mod.set_param_hint('area', value=100, vary=True, min=0,
                                                 expr=str(ename)+'_ka1_'+'area')
                    gauss_mod.set_param_hint('center', value=val, vary=False)
                    gauss_mod.set_param_hint('sigma', value=1, vary=False)
                    ratio_v = e.cs(incident_energy)[line_name]/e.cs(incident_energy)['ka1']
                    gauss_mod.set_param_hint('ratio',
                                             value=ratio_v, vary=False)

                    if ename in element_adjust:
                        if parameter['element'][ename].has_key(line_name.lower()+'_position'):
                            pos_val = parameter['element'][ename][line_name.lower()+'_position']
                            if pos_val != 0:
                                gauss_mod.set_param_hint('center', value=val, vary=True,
                                                         min=val*(1-pos_val), max=val*(1+pos_val))
                                logger.warning('change element {0} {1} postion '
                                               'within range {2}'.format(ename, line_name, [-pos_val, pos_val]))

                        if parameter['element'][ename].has_key(line_name.lower()+'_width'):
                            width_val = parameter['element'][ename][line_name.lower()+'_width']
                            if width_val != 0:
                                gauss_mod.set_param_hint('sigma', value=1, vary=True,
                                                         min=1-width_val, max=1+width_val)
                                logger.warning('change element {0} {1} peak width '
                                               'within range {2}'.format(ename, line_name, [-width_val, width_val]))

                    mod = mod + gauss_mod
                logger.debug('Finished building element peak for {0}'.format(ename))

            elif ename in l_line:
                ename = ename[:-2]
                e = Element(ename)
                if e.cs(incident_energy)['la1'] == 0:
                    logger.info('{0} La1 emission line is not activated '
                                'at this energy {1}'.format(ename, incident_energy))
                    continue

                # L lines
                #gauss_mod = GaussModel_Llines(prefix=str(ename)+'_l_line_')

                #gauss_mod.set_param_hint('area', value=100, vary=True, min=0)
                #gauss_mod.set_param_hint('fwhm_offset', value=0.1, vary=True, expr='fwhm_offset')
                #gauss_mod.set_param_hint('fwhm_fanoprime', value=0.1, vary=True, expr='fwhm_fanoprime')

                for num, item in enumerate(e.emission_line.all[4:-4]):

                    line_name = item[0]
                    val = item[1]

                    if e.cs(incident_energy)[line_name] == 0:
                        continue

                    gauss_mod = GaussModel_xrf(prefix=str(ename)+'_'+str(line_name)+'_')

                    gauss_mod.set_param_hint('fwhm_offset', value=0.1, vary=True, expr='fwhm_offset')
                    gauss_mod.set_param_hint('fwhm_fanoprime', value=0.1, vary=True, expr='fwhm_fanoprime')

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

                    mod = mod + gauss_mod

        self.mod = mod
        return

    def model_fit(self, x, y, w=None, method='leastsq', **kws):
        """
        Parameters
        ----------
        
        """

        self.model_spectrum()

        print ("fitting params: ", kws)

        pars = self.mod.make_params()
        result = self.mod.fit(y, pars, x=x, weights=w,
                              method=method, fit_kws=kws)
        return result
