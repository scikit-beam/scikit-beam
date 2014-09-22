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
import six
import numpy as np
import sys
import inspect
import json
import os


import logging
logger = logging.getLogger(__name__)

from skxray.fitting.model.physics_peak import (elastic_peak, compton_peak,
                                              gauss_peak)

from skxray.fitting.base.parameter_data import get_para
from skxray.constants import Element

from lmfit import Model
from lmfit.models import GaussianModel as LmGaussianModel
from lmfit.models import LorentzianModel as LmLorentzianModel
from .lineshapes import (elastic, compton, gaussian_tail, gausssian_step,
                         lorentzian2, gaussian, lorentzian
)

import logging
logger = logging.getLogger(__name__)

from skxray.fitting.lineshapes import (elastic, compton, gaussian,
                                       lorentzian, lorentzian2)

from skxray.fitting.base.parameter_data import get_para


from lmfit import Model



k_line = ['Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
          'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
          'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
          'In', 'Sn', 'Sb', 'Te', 'I', 'dummy', 'dummy']

l_line = ['Mo_L', 'Tc_L', 'Ru_L', 'Rh_L', 'Pd_L', 'Ag_L', 'Cd_L', 'In_L', 'Sn_L', 'Sb_L', 'Te_L', 'I_L', 'Xe_L', 'Cs_L', 'Ba_L', 'La_L', 'Ce_L', 'Pr_L', 'Nd_L', 'Pm_L', 'Sm_L',
          'Eu_L', 'Gd_L', 'Tb_L', 'Dy_L', 'Ho_L', 'Er_L', 'Tm_L', 'Yb_L', 'Lu_L', 'Hf_L', 'Ta_L', 'W_L', 'Re_L', 'Os_L', 'Ir_L', 'Pt_L', 'Au_L', 'Hg_L', 'Tl_L',
          'Pb_L', 'Bi_L', 'Po_L', 'At_L', 'Rn_L', 'Fr_L', 'Ac_L', 'Th_L', 'Pa_L', 'U_L', 'Np_L', 'Pu_L', 'Am_L', 'Br_L', 'Ga_L']

m_line = ['Au_M', 'Pb_M', 'U_M', 'noise', 'Pt_M', 'Ti_M', 'Gd_M', 'dummy', 'dummy']


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


# SUBCLASS LMFIT MODELS TO REWRITE THE DOCS
class GaussianModel(LmGaussianModel):
    __doc__ = _gen_class_docs(gaussian)

    def __init__(self, *args, **kwargs):
         super(GaussianModel, self).__init__(*args, **kwargs)


class LorentzianModel(LmLorentzianModel):
    __doc__ = _gen_class_docs(lorentzian)

    def __init__(self, *args, **kwargs):
        super(LorentzianModel, self).__init__(*args, **kwargs)


# DEFINE NEW MODELS
class ElasticModel(Model):

    __doc__ = _gen_class_docs(elastic)

    def __init__(self, *args, **kwargs):
        super(ElasticModel, self).__init__(elastic, *args, **kwargs)
        set_default(self, elastic)
        self.set_param_hint('epsilon', value=2.96, vary=False)


class ComptonModel(Model):

    __doc__ = _gen_class_docs(compton)

    def __init__(self, *args, **kwargs):
        super(ComptonModel, self).__init__(compton, *args, **kwargs)
        set_default(self, compton)
        self.set_param_hint('epsilon', value=2.96, vary=False)


class Lorentzian2Model(Model):

    __doc__ = _gen_class_docs(lorentzian2)

    def __init__(self, *args, **kwargs):
        super(Lorentzian2Model, self).__init__(lorentzian2, *args, **kwargs)



def gausspeak_k_lines(x, area,
                      fwhm_offset,
                      fwhm_fanoprime,
                      center1, sigma1, ratio1,
                      center2, sigma2, ratio2,
                      center3, sigma3, ratio3,
                      center4, sigma4, ratio4):

    def get_sigma(center):
        return np.sqrt((fwhm_offset/2.3548)**2 + center*2.96*fwhm_fanoprime)

    g1 = gauss_peak(x, area, center1, sigma1*get_sigma(center1)) * ratio1
    g2 = gauss_peak(x, area, center2, sigma2*get_sigma(center2)) * ratio2
    g3 = gauss_peak(x, area, center3, sigma3*get_sigma(center3)) * ratio3
    g4 = gauss_peak(x, area, center4, sigma4*get_sigma(center4)) * ratio4

    return g1 + g2 + g3 + g4


class GaussModel_Klines(Model):

    #__doc__ = _gen_class_docs(gausspeak_k_lines)

    def __init__(self, *args, **kwargs):
        super(GaussModel_Klines, self).__init__(gausspeak_k_lines, *args, **kwargs)


def gausspeak_l_lines(x, area,
                      fwhm_offset,
                      fwhm_fanoprime,
                      center1, sigma1, ratio1,
                      center2, sigma2, ratio2,
                      center3, sigma3, ratio3,
                      center4, sigma4, ratio4,
                      center5, sigma5, ratio5,
                      center6, sigma6, ratio6,
                      center7, sigma7, ratio7,
                      center8, sigma8, ratio8,
                      center9, sigma9, ratio9,
                      center10, sigma10, ratio10,
                      center11, sigma11, ratio11,
                      center12, sigma12, ratio12,
                      center13, sigma13, ratio13):

    def get_sigma(center):
        return np.sqrt((fwhm_offset/2.3548)**2 + center*2.96*fwhm_fanoprime)

    g1 = gauss_peak(x, area, center1, sigma1*get_sigma(center1)) * ratio1
    g2 = gauss_peak(x, area, center2, sigma2*get_sigma(center2)) * ratio2
    g3 = gauss_peak(x, area, center3, sigma3*get_sigma(center3)) * ratio3
    g4 = gauss_peak(x, area, center4, sigma4*get_sigma(center4)) * ratio4
    g5 = gauss_peak(x, area, center5, sigma5*get_sigma(center5)) * ratio5
    g6 = gauss_peak(x, area, center6, sigma6*get_sigma(center6)) * ratio6
    g7 = gauss_peak(x, area, center7, sigma7*get_sigma(center7)) * ratio7
    g8 = gauss_peak(x, area, center8, sigma8*get_sigma(center8)) * ratio8
    g9 = gauss_peak(x, area, center9, sigma9*get_sigma(center9)) * ratio9
    g10 = gauss_peak(x, area, center10, sigma10*get_sigma(center10)) * ratio10
    g11 = gauss_peak(x, area, center11, sigma11*get_sigma(center11)) * ratio11
    g12 = gauss_peak(x, area, center12, sigma12*get_sigma(center12)) * ratio12
    g13 = gauss_peak(x, area, center13, sigma13*get_sigma(center13)) * ratio13

    return g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 + g9 + g10 + g11 + g12 + g13


class GaussModel_Llines(Model):

    #__doc__ = _gen_class_docs(gausspeak_k_lines)

    def __init__(self, *args, **kwargs):
        super(GaussModel_Llines, self).__init__(gausspeak_l_lines, *args, **kwargs)


def gauss_peak_xrf(x, area, center, sigma, ratio, fwhm_offset, fwhm_fanoprime):

    def get_sigma(center):
        return np.sqrt((fwhm_offset/2.3548)**2 + center*2.96*fwhm_fanoprime)

    return gauss_peak(x, area, center, sigma*get_sigma(center)) * ratio


class GaussModel_xrf(Model):

    #__doc__ = _gen_class_docs(gausspeak_k_lines)

    def __init__(self, *args, **kwargs):
        super(GaussModel_xrf, self).__init__(gauss_peak_xrf, *args, **kwargs)


def _set_value(para_name, input_dict, model_name):

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

        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 config_file)

        json_data = open(file_path, 'r')
        self.parameter = json.load(json_data)

        self.element_list = self.parameter['element_list'].split()
        self.incident_energy = self.parameter['coherent_sct_energy']['value']

        self.parameter_default = get_para()

        return

    def setComptonModel(self):
        """
        setup parameters related to Compton model
        """
        compton = ComptonModel()

        compton_list = ['coherent_sct_energy','compton_amplitude',
                        'compton_angle', 'fwhm_offset', 'fwhm_fanoprime',
                        'compton_gamma', 'compton_f_tail',
                        'compton_f_step', 'compton_fwhm_corr',
                        'compton_hi_gamma', 'compton_hi_f_tail']

        for name in compton_list:
            if name in self.parameter.keys():
                _set_value(name, self.parameter[name], compton)
            else:
                _set_value(name, self.parameter_default[name], compton)

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

        # set constrains for the following global parameters
        elastic.set_param_hint(name='fwhm_offset', value=0.1, vary=True, expr='fwhm_offset')
        elastic.set_param_hint(name='fwhm_fanoprime', value=0.0, vary=True, expr='fwhm_fanoprime')
        elastic.set_param_hint(name='coherent_sct_energy', value=self.incident_energy,
                               expr='coherent_sct_energy')

        return elastic

    def model_spectrum(self):

        incident_energy = self.incident_energy
        element_list = self.element_list
        parameter = self.parameter

        mod = self.setComptonModel() + self.setElasticModel()
        #mod = self.setElasticModel()

        if parameter.has_key('element'):
            element_adjust = [item['name'] for item in parameter['element']]
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

                # K lines
                # It is much faster to construct only one model
                # to construct four gauss models with constrains
                # relating each other.
                #gauss_mod = GaussModel_Klines(prefix=str(ename)+'_k_line_')

                #gauss_mod.set_param_hint('area', value=100, vary=True, min=0)
                #gauss_mod.set_param_hint('fwhm_offset', value=0.1, vary=True, expr='fwhm_offset')
                #gauss_mod.set_param_hint('fwhm_fanoprime', value=0.1, vary=True, expr='fwhm_fanoprime')

                for num, item in enumerate(e.emission_line.all[:4]):
                    #val = e.emission_line['ka1']
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
                    print (self.parameter['element'])
                    gauss_mod.set_param_hint('center', value=val, vary=False)
                    gauss_mod.set_param_hint('sigma', value=1, vary=False)

                    ratio_v = e.cs(incident_energy)[line_name]/e.cs(incident_energy)['ka1']
                    gauss_mod.set_param_hint('ratio',
                                             value=ratio_v, vary=False)

                    #gauss_mod.set_param_hint('center'+str(num+1), value=val, vary=False)
                    #gauss_mod.set_param_hint('sigma'+str(num+1), value=1, vary=False)
                    #print ("value is", e.cs(incident_energy)['ka1'])
                    #ratio_v = e.cs(incident_energy)[line_name]/e.cs(incident_energy)['ka1']
                    #print ("ratio is", ratio_v)
                    #if ratio_v == 0 or ratio_v == 1:
                    #    gauss_mod.set_param_hint('ratio'+str(num+1),
                    #                             value=ratio_v, vary=False)
                    #else:
                    #    gauss_mod.set_param_hint('ratio'+str(num+1),
                    #                             value=ratio_v, vary=False)
                                                 #min=ratio_v*0.8, max=ratio_v*1.2)

                    mod = mod + gauss_mod

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
                    #gauss_mod.set_param_hint('ratio_val',
                    #                         value=e.cs(incident_energy)[line_name]/e.cs(incident_energy)['la1'])

                    if line_name == 'la1':
                        gauss_mod.set_param_hint('area', value=100, vary=True)
                                             #expr=gauss_mod.prefix+'ratio_val * '+str(ename)+'_la1_'+'area')
                    else:
                        gauss_mod.set_param_hint('area', value=100, vary=True,
                                                 expr=str(ename)+'_la1_'+'area')

                    #    gauss_mod.set_param_hint('area', value=100, vary=True,
                    #                             expr=gauss_mod.prefix+'ratio_val * '+str(ename)+'_la1_'+'area')

                    gauss_mod.set_param_hint('center', value=val, vary=False)
                    gauss_mod.set_param_hint('sigma', value=1, vary=False)
                    gauss_mod.set_param_hint('ratio',
                                             value=e.cs(incident_energy)[line_name]/e.cs(incident_energy)['la1'],
                                             vary=False)

                    #gauss_mod.set_param_hint('center'+str(num+1), value=val, vary=False)
                    #gauss_mod.set_param_hint('sigma'+str(num+1), value=1, vary=False)
                    #print ("value is", e.cs(incident_energy)['ka1'])
                    #ratio_v = e.cs(incident_energy)[line_name]/e.cs(incident_energy)['la1']
                    #print ("ratio is", ratio_v)
                    #if ratio_v == 0 or ratio_v == 1:
                    #    gauss_mod.set_param_hint('ratio'+str(num+1),
                    #                             value=ratio_v, vary=False)
                    #else:
                    #    gauss_mod.set_param_hint('ratio'+str(num+1),
                    #                             value=ratio_v, vary=False)
                                                 #min=ratio_v*0.8, max=ratio_v*1.2)

                    mod = mod + gauss_mod

        self.mod = mod
        return


    def model_fit(self, x, y, w=None):
        self.model_spectrum()
        result = self.mod.fit(y, x=x, weights=w)
        return result

    def get_bg(self, y):
        """snip method to get background"""
        return snip_method(y, 0, 0.01, 0)

