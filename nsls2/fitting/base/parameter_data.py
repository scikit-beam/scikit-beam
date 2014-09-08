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

from __future__ import (absolute_import, division, unicode_literals, print_function)
import six

"""
Parameter dictionary are included for xrf fitting.
Element data not included.

Some parameters are defined as

use :
fixed: value is fixed
lohi: with both low and high boundary
lo: with low boundary
hi: with high boundary
none: no fitting boundary


Different fitting strategies are included to turn on or turn off some parameters.
Those strategies are default, linear, free_energy, free_all and e_calibration.
They are empirical experience from authors of the original code.
"""


para_dict = {'coherent_sct_amplitude ': {'use': 'none', 'min': 7.0, 'max': 8.0, 'value': 6.0, 'e_calibration': 'none', 'free_energy': 'none', 'free_all': 'none', 'default': 'none', 'linear': 'none'},
             'coherent_sct_energy ': {'use': 'none', 'min': 10.4, 'max': 12.4, 'value': 11.8, 'e_calibration': 'fixed', 'free_energy': 'lohi', 'free_all': 'lohi', 'default': 'fixed', 'linear': 'fixed'},
             'compton_amplitude ': {'use': 'none', 'min': 0.0, 'max': 10.0, 'value': 5.0, 'e_calibration': 'none', 'free_energy': 'none', 'free_all': 'none', 'default': 'none', 'linear': 'none'},
             'compton_angle ': {'use': 'lohi', 'min': 75.0, 'max': 90.0, 'value': 90.0, 'e_calibration': 'fixed', 'free_energy': 'lohi', 'free_all': 'lohi', 'default': 'fixed', 'linear': 'fixed'},
             'compton_f_step ': {'use': 'lohi', 'min': 0.0, 'max': 1.5, 'value': 0.1, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'lohi', 'default': 'fixed', 'linear': 'fixed'},
             'compton_f_tail ': {'use': 'lohi', 'min': 0.0, 'max': 3.0, 'value': 0.8, 'e_calibration': 'fixed', 'free_energy': 'lo', 'free_all': 'fixed', 'default': 'fixed', 'linear': 'fixed'},
             'compton_fwhm_corr ': {'use': 'lohi', 'min': 0.1, 'max': 3.0, 'value': 1.4, 'e_calibration': 'fixed', 'free_energy': 'lohi', 'free_all': 'lohi', 'default': 'fixed', 'linear': 'fixed'},
             'compton_gamma ': {'use': 'none', 'min': 0.1, 'max': 10.0, 'value': 1.0, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'lohi', 'default': 'fixed', 'linear': 'fixed'},
             'compton_hi_f_tail ': {'use': 'none', 'min': 1e-06, 'max': 1.0, 'value': 0.01, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'fixed', 'default': 'fixed', 'linear': 'fixed'},
             'compton_hi_gamma ': {'use': 'none', 'min': 0.1, 'max': 3.0, 'value': 1.0, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'fixed', 'default': 'fixed', 'linear': 'fixed'},
             'e_e_calibration ': {'use': 'fixed', 'min': 0.001, 'max': 0.1, 'value': 1.0, 'e_calibration': 'lohi', 'free_energy': 'lohi', 'free_all': 'lohi', 'default': 'fixed', 'linear': 'fixed'},
             'e_offset ': {'use': 'fixed', 'min': -0.2, 'max': 0.2, 'value': 0.0, 'e_calibration': 'lohi', 'free_energy': 'lohi', 'free_all': 'lohi', 'default': 'fixed', 'linear': 'fixed'},
             'e_quadratic ': {'use': 'none', 'min': -0.0001, 'max': 0.0001, 'value': 0.0, 'e_calibration': 'lohi', 'free_energy': 'lohi', 'free_all': 'lohi', 'default': 'fixed', 'linear': 'fixed'},
             'f_step_e_calibration ': {'use': 'none', 'min': 0.0, 'max': 1.0, 'value': 0.0, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'fixed', 'default': 'fixed', 'linear': 'fixed'},
             'f_step_offset ': {'use': 'none', 'min': 0.0, 'max': 1.0, 'value': 0.0, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'fixed', 'default': 'fixed', 'linear': 'fixed'},
             'f_step_quadratic ': {'use': 'none', 'min': 0.0, 'max': 0.0, 'value': 0.0, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'fixed', 'default': 'fixed', 'linear': 'fixed'},
             'f_tail_e_calibration ': {'use': 'none', 'min': 0.0, 'max': 1.0, 'value': 0.01, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'lohi', 'default': 'fixed', 'linear': 'fixed'},
             'f_tail_offset ': {'use': 'none', 'min': 0.0, 'max': 0.1, 'value': 0.04, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'lohi', 'default': 'fixed', 'linear': 'fixed'},
             'f_tail_quadratic ': {'use': 'none', 'min': 0.0, 'max': 0.01, 'value': 0.0, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'fixed', 'default': 'fixed', 'linear': 'fixed'},
             'fwhm_fanoprime ': {'use': 'lohi', 'min': 1e-06, 'max': 0.05, 'value': 0.00012, 'e_calibration': 'fixed', 'free_energy': 'lohi', 'free_all': 'lohi', 'default': 'fixed', 'linear': 'fixed'},
             'fwhm_offset ': {'use': 'lohi', 'min': 0.005, 'max': 0.5, 'value': 0.12, 'e_calibration': 'fixed', 'free_energy': 'lohi', 'free_all': 'lohi', 'default': 'fixed', 'linear': 'fixed'},
             'gamma_e_calibration ': {'use': 'none', 'min': 0.0, 'max': 3.0, 'value': 0.0, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'fixed', 'default': 'fixed', 'linear': 'fixed'},
             'gamma_offset ': {'use': 'none', 'min': 0.1, 'max': 10.0, 'value': 2.0, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'fixed', 'default': 'fixed', 'linear': 'fixed'},
             'gamma_quadratic ': {'use': 'none', 'min': 0.0, 'max': 0.0, 'value': 0.0, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'fixed', 'default': 'fixed', 'linear': 'fixed'},
             'ge_escape ': {'use': 'none', 'min': 0.0, 'max': 1.0, 'value': 0.0, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'fixed', 'default': 'fixed', 'linear': 'fixed'},
             'kb_f_tail_e_calibration ': {'use': 'none', 'min': 0.0, 'max': 0.02, 'value': 0.0, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'lohi', 'default': 'fixed', 'linear': 'fixed'},
             'kb_f_tail_offset ': {'use': 'none', 'min': 0.0, 'max': 0.2, 'value': 0.0, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'lohi', 'default': 'fixed', 'linear': 'fixed'},
             'kb_f_tail_quadratic ': {'use': 'none', 'min': 0.0, 'max': 0.0, 'value': 0.0, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'fixed', 'default': 'fixed', 'linear': 'fixed'},
             'e_calibration ': {'use': 'none', 'min': 0.0, 'max': 1.0, 'value': 0.0, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'fixed', 'default': 'fixed', 'linear': 'fixed'},
             'pileup0 ': {'use': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'fixed', 'default': 'fixed', 'linear': 'fixed'},
             'pileup1 ': {'use': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'fixed', 'default': 'fixed', 'linear': 'fixed'},
             'pileup2 ': {'use': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'fixed', 'default': 'fixed', 'linear': 'fixed'},
             'pileup3 ': {'use': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'fixed', 'default': 'fixed', 'linear': 'fixed'},
             'pileup4 ': {'use': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'fixed', 'default': 'fixed', 'linear': 'fixed'},
             'pileup5 ': {'use': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'fixed', 'default': 'fixed', 'linear': 'fixed'},
             'pileup6 ': {'use': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'fixed', 'default': 'fixed', 'linear': 'fixed'},
             'pileup7 ': {'use': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'fixed', 'default': 'fixed', 'linear': 'fixed'},
             'pileup8 ': {'use': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'fixed', 'default': 'fixed', 'linear': 'fixed'},
             'si_escape ': {'use': 'none', 'min': 0.0, 'max': 0.5, 'value': 0.0, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'fixed', 'default': 'fixed', 'linear': 'fixed'},
             'snip_width ': {'use': 'none', 'min': 0.1, 'max': 2.82842712475, 'value': 0.15, 'e_calibration': 'fixed', 'free_energy': 'fixed', 'free_all': 'fixed', 'default': 'fixed', 'linear': 'fixed'},
             }
