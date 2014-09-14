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



kele = ['Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
        'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
        'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'In', 'Sn', 'Sb', 'Te', 'I', 'dummy', 'dummy']

#lele = ['Mo_L', 'Ag_L', 'Sn_L', 'Cd_L', 'I_L', 'Cs_L', 'Ba_L', 'Eu_L', 'Gd_L', 'W_L', 'Pt_L', 'Au_L',
#        'Hg_L', 'Pb_L', 'U_L', 'Pu_L', 'Sm_L', 'Y_L', 'Pr_L', 'Ce_L', 'Zr_L', 'Os_L', 'Rb_L', 'Ru_L']

lele = ['Mo_L', 'Tc_L', 'Ru_L', 'Rh_L', 'Pd_L', 'Ag_L', 'Cd_L', 'In_L', 'Sn_L', 'Sb_L', 'Te_L', 'I_L', 'Xe_L', 'Cs_L', 'Ba_L', 'La_L', 'Ce_L', 'Pr_L', 'Nd_L', 'Pm_L', 'Sm_L',
        'Eu_L', 'Gd_L', 'Tb_L', 'Dy_L', 'Ho_L', 'Er_L', 'Tm_L', 'Yb_L', 'Lu_L', 'Hf_L', 'Ta_L', 'W_L', 'Re_L', 'Os_L', 'Ir_L', 'Pt_L', 'Au_L', 'Hg_L', 'Tl_L',
        'Pb_L', 'Bi_L', 'Po_L', 'At_L', 'Rn_L', 'Fr_L', 'Ac_L', 'Th_L', 'Pa_L', 'U_L', 'Np_L', 'Pu_L', 'Am_L', 'Br_L', 'Ga_L']

mele = ['Au_M', 'Pb_M', 'U_M', 'noise', 'Pt_M', 'Ti_M', 'Gd_M', 'dummy', 'dummy']


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
        self.set_param_hint('matrix', value=False, vary=False)


class Lorentzian2Model(Model):

    __doc__ = _gen_class_docs(lorentzian2)

    def __init__(self, *args, **kwargs):
        super(Lorentzian2Model, self).__init__(lorentzian2, *args, **kwargs)




class ModelSpectrum(object):

    def __init__(self, incident_energy, element_list):
        self.incident_energy = incident_energy
        self.element_list = element_list
        return


    def setComptonModel(self):
        """
        need to read input file to setup parameters
        """
        compton = ComptonModel()
        # parameters not sensitive
        compton.set_param_hint(name='compton_hi_gamma', value=0.25, vary=False)#min=0.0, max=4.0)
        compton.set_param_hint(name='fwhm_offset', value=0.1, vary=True, expr='e_fwhm_offset')

        # parameters with boundary
        compton.set_param_hint(name='coherent_sct_energy', value=11.78, vary=True, min=11.77, max=11.79)
        compton.set_param_hint(name='compton_gamma', value=5.2, vary=True, min=1, max=10.5)
        compton.set_param_hint(name='compton_f_tail', value=0.5, vary=True, min=0, max=2.0)
        compton.set_param_hint(name='compton_hi_gamma', value=0.2, min=1, max=2.5, vary=True)
        compton.set_param_hint(name='compton_hi_f_tail', value=0.005, vary=False)#min=0, max=0.05)
        compton.set_param_hint(name='compton_fwhm_corr', value=3.5, min=2.0, max=4.5)
        compton.set_param_hint(name='compton_amplitude', value=80000)
        compton.set_param_hint(name='compton_angle', value=90, vary=True)
        compton.set_param_hint(name='matrix', value=True, vary=False)

        return compton


    def setElasticModel(self):
        """
        need to read input file to setup parameters
        """
        elastic = ElasticModel(prefix='e_')

        # fwhm_offset is not a sensitive parameter, used as a fixed value
        elastic.set_param_hint(name='fwhm_offset', value=0.1, vary=True)
        elastic.set_param_hint(name='coherent_sct_energy', value=11.78, expr='coherent_sct_energy')# min=11.77, max=11.79)
        elastic.set_param_hint(name='coherent_sct_amplitude', value=50000)

        return elastic


    def model_spectrum(self):

        incident_energy = self.incident_energy
        element_list = self.element_list

        mod = self.setComptonModel() + self.setElasticModel()


        #e = Element('Si')

        for ename in element_list:
            if ename not in kele:
                continue
            #print (ename)
            e = Element(ename)
            if e.cs(incident_energy)['ka1'] == 0:
                logger.info('{0} Ka emission line is not activated '
                            'at this energy {1}'.format(ename, incident_energy))
                continue

            # k lines
            #for i in np.arange(1): #e.emission_line.all[:4]:
            val = e.emission_line['ka1']
            gauss_mod = GaussModel(prefix=str(ename) + '_')
            gauss_mod.set_param_hint('area', value=100, vary=True)
            gauss_mod.set_param_hint('center', value=val, vary=False)
            gauss_mod.set_param_hint('sigma', value=0.05, vary=False)
            mod = mod + gauss_mod

        self.mod = mod
        return


    def model_fit(self, x, y):
        self.model_spectrum()
        #print (self.mod.param_names)
        #p = self.mod.make_params()
        result = self.mod.fit(y, x=x)
        return result




