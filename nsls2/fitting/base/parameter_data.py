# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# @author: Li Li (lili@bnl.gov)                                        #
# created on 08/28/2014                                                #
#                                                                      #
# Original code:                                                       #
# @author: Mirna Lerotic, 2nd Look Consulting                          #
#         http://www.2ndlookconsulting.com/                            #
# Copyright (c) 2013, Stefan Vogt, Argonne National Laboratory         #
# All rights reserved.                                                 #
#                                                                      #
# Redistribution and bound_type in source and binary forms, with or without   #
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
#   National Laboratory nor the names of its contributors may be bound_typed  #
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
# SERVICES; LOSS OF bound_type, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)   #
# HOWEVER CAbound_typeD AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,  #
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OTHERWISE) ARISING   #
# IN ANY WAY OUT OF THE bound_type OF THIS SOFTWARE, EVEN IF ADVISED OF THE   #
# POSSIBILITY OF SUCH DAMAGE.                                          #
########################################################################

from __future__ import (absolute_import, division, unicode_literals, print_function)
import six


"""
Parameter dictionary are included for xrf fitting.
Element data not included.

Some parameters are defined as

bound_type :
fixed: value is fixed
lohi: with both low and high boundary
lo: with low boundary
hi: with high boundary
none: no fitting boundary


Different fitting strategies are included to turn on or turn off some parameters.
Those strategies are default, linear, free_energy, free_all and e_calibration.
They are empirical experience from authors of the original code.
"""


para_dict = {'coherent_sct_amplitude': {'bound_type': 'none', 'min': 7.0, 'max': 8.0, 'value': 6.0},
             'coherent_sct_energy': {'bound_type': 'none', 'min': 10.4, 'max': 12.4, 'value': 11.8},
             'compton_amplitude': {'bound_type': 'none', 'min': 0.0, 'max': 10.0, 'value': 5.0},
             'compton_angle': {'bound_type': 'lohi', 'min': 75.0, 'max': 90.0, 'value': 90.0},
             'compton_f_step': {'bound_type': 'lohi', 'min': 0.0, 'max': 1.5, 'value': 0.1},
             'compton_f_tail': {'bound_type': 'lohi', 'min': 0.0, 'max': 3.0, 'value': 0.8},
             'compton_fwhm_corr': {'bound_type': 'lohi', 'min': 0.1, 'max': 3.0, 'value': 1.4},
             'compton_gamma': {'bound_type': 'none', 'min': 0.1, 'max': 10.0, 'value': 1.0},
             'compton_hi_f_tail': {'bound_type': 'none', 'min': 1e-06, 'max': 1.0, 'value': 0.01},
             'compton_hi_gamma': {'bound_type': 'none', 'min': 0.1, 'max': 3.0, 'value': 1.0},
             'e_linear': {'bound_type': 'fixed', 'min': 0.001, 'max': 0.1, 'value': 1.0},
             'e_offset': {'bound_type': 'fixed', 'min': -0.2, 'max': 0.2, 'value': 0.0},
             'e_quadratic': {'bound_type': 'none', 'min': -0.0001, 'max': 0.0001, 'value': 0.0},
             'f_step_linear': {'bound_type': 'none', 'min': 0.0, 'max': 1.0, 'value': 0.0},
             'f_step_offset': {'bound_type': 'none', 'min': 0.0, 'max': 1.0, 'value': 0.0},
             'f_step_quadratic': {'bound_type': 'none', 'min': 0.0, 'max': 0.0, 'value': 0.0},
             'f_tail_linear': {'bound_type': 'none', 'min': 0.0, 'max': 1.0, 'value': 0.01},
             'f_tail_offset': {'bound_type': 'none', 'min': 0.0, 'max': 0.1, 'value': 0.04},
             'f_tail_quadratic': {'bound_type': 'none', 'min': 0.0, 'max': 0.01, 'value': 0.0},
             'fwhm_fanoprime': {'bound_type': 'lohi', 'min': 1e-06, 'max': 0.05, 'value': 0.00012},
             'fwhm_offset': {'bound_type': 'lohi', 'min': 0.005, 'max': 0.5, 'value': 0.12},
             'gamma_linear': {'bound_type': 'none', 'min': 0.0, 'max': 3.0, 'value': 0.0},
             'gamma_offset': {'bound_type': 'none', 'min': 0.1, 'max': 10.0, 'value': 2.0},
             'gamma_quadratic': {'bound_type': 'none', 'min': 0.0, 'max': 0.0, 'value': 0.0},
             'ge_escape': {'bound_type': 'none', 'min': 0.0, 'max': 1.0, 'value': 0.0},
             'kb_f_tail_linear': {'bound_type': 'none', 'min': 0.0, 'max': 0.02, 'value': 0.0},
             'kb_f_tail_offset': {'bound_type': 'none', 'min': 0.0, 'max': 0.2, 'value': 0.0},
             'kb_f_tail_quadratic': {'bound_type': 'none', 'min': 0.0, 'max': 0.0, 'value': 0.0},
             'linear': {'bound_type': 'none', 'min': 0.0, 'max': 1.0, 'value': 0.0},
             'pileup0': {'bound_type': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10},
             'pileup1': {'bound_type': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10},
             'pileup2': {'bound_type': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10},
             'pileup3': {'bound_type': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10},
             'pileup4': {'bound_type': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10},
             'pileup5': {'bound_type': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10},
             'pileup6': {'bound_type': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10},
             'pileup7': {'bound_type': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10},
             'pileup8': {'bound_type': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10},
             'si_escape': {'bound_type': 'none', 'min': 0.0, 'max': 0.5, 'value': 0.0},
             'snip_width': {'bound_type': 'none', 'min': 0.1, 'max': 2.82842712475, 'value': 0.15},
             }


# fitting strategy
linear = {'coherent_sct_amplitude': 'none', 'compton_amplitude': 'none'}

e_calibration = {'coherent_sct_amplitude': 'none', 'compton_amplitude': 'none',
                 'e_linear': 'lohi', 'e_offset': 'lohi', 'e_quadratic': 'lohi'}

free_energy = {'coherent_sct_amplitude': 'none', 'coherent_sct_energy': 'lohi',
               'compton_amplitude': 'none', 'compton_angle': 'lohi',
               'compton_f_tail': 'lo', 'compton_fwhm_corr': 'lohi',
               'e_linear': 'lohi', 'e_offset': 'lohi', 'e_quadratic': 'lohi',
               'fwhm_fanoprime': 'lohi', 'fwhm_offset': 'lohi'}

free_all = {'coherent_sct_amplitude': 'none', 'coherent_sct_energy': 'lohi',
            'compton_amplitude': 'none', 'compton_angle': 'lohi',
            'compton_f_step': 'lohi', 'compton_fwhm_corr': 'lohi', 'compton_gamma': 'lohi',
            'e_linear': 'lohi', 'e_offset': 'lohi', 'e_quadratic': 'lohi',
            'f_tail_linear': 'lohi', 'f_tail_offset': 'lohi',
            'fwhm_fanoprime': 'lohi', 'fwhm_offset': 'lohi',
            'kb_f_tail_linear': 'lohi', 'kb_f_tail_offset': 'lohi'}
