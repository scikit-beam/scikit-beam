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

from __future__ import (absolute_import, division)
import six

"""
Parameter dictionary are included for xrf fitting.
Element data not included.

Some parameters are defined as

use :
1 fixed
2 with both low and high boundary
3 with low boundary
4 with high boundary
5 no fitting boundary

option0, option1, ..., option4:
Those are different strategies to turn on or turn off some parameters.
They are empirical experience from authors of the original code.
"""

para_dict = {'coherent_sct_amplitude ': {'use': 5.0, 'min': 7.0, 'max': 8.0, 'value': 6.0, 'option4': 5.0, 'option2': 5.0, 'option3': 5.0, 'option0': 5.0, 'option1': 5.0},
             'coherent_sct_energy ': {'use': 1.0, 'min': 10.4, 'max': 12.4, 'value': 11.8, 'option4': 1.0, 'option2': 2.0, 'option3': 2.0, 'option0': 1.0, 'option1': 1.0},
             'compton_amplitude ': {'use': 5.0, 'min': 0.0, 'max': 10.0, 'value': 5.0, 'option4': 5.0, 'option2': 5.0, 'option3': 5.0, 'option0': 5.0, 'option1': 5.0},
             'compton_angle ': {'use': 2.0, 'min': 75.0, 'max': 90.0, 'value': 90.0, 'option4': 1.0, 'option2': 2.0, 'option3': 2.0, 'option0': 1.0, 'option1': 1.0},
             'compton_f_step ': {'use': 2.0, 'min': 0.0, 'max': 1.5, 'value': 0.1, 'option4': 1.0, 'option2': 1.0, 'option3': 2.0, 'option0': 1.0, 'option1': 1.0},
             'compton_f_tail ': {'use': 2.0, 'min': 0.0, 'max': 3.0, 'value': 0.8, 'option4': 1.0, 'option2': 3.0, 'option3': 1.0, 'option0': 1.0, 'option1': 1.0},
             'compton_fwhm_corr ': {'use': 2.0, 'min': 0.1, 'max': 3.0, 'value': 1.4, 'option4': 1.0, 'option2': 2.0, 'option3': 2.0, 'option0': 1.0, 'option1': 1.0},
             'compton_gamma ': {'use': 1.0, 'min': 0.1, 'max': 10.0, 'value': 1.0, 'option4': 1.0, 'option2': 1.0, 'option3': 2.0, 'option0': 1.0, 'option1': 1.0},
             'compton_hi_f_tail ': {'use': 1.0, 'min': 1e-06, 'max': 1.0, 'value': 0.01, 'option4': 1.0, 'option2': 1.0, 'option3': 1.0, 'option0': 1.0, 'option1': 1.0},
             'compton_hi_gamma ': {'use': 1.0, 'min': 0.1, 'max': 3.0, 'value': 1.0, 'option4': 1.0, 'option2': 1.0, 'option3': 1.0, 'option0': 1.0, 'option1': 1.0},
             'e_linear ': {'use': 0.0, 'min': 0.001, 'max': 0.1, 'value': 1.0, 'option4': 2.0, 'option2': 2.0, 'option3': 2.0, 'option0': 1.0, 'option1': 1.0},
             'e_offset ': {'use': 0.0, 'min': -0.2, 'max': 0.2, 'value': 0.0, 'option4': 2.0, 'option2': 2.0, 'option3': 2.0, 'option0': 1.0, 'option1': 1.0},
             'e_quadratic ': {'use': 1.0, 'min': -0.0001, 'max': 0.0001, 'value': 0.0, 'option4': 2.0, 'option2': 2.0, 'option3': 2.0, 'option0': 1.0, 'option1': 1.0},
             'f_step_linear ': {'use': 1.0, 'min': 0.0, 'max': 1.0, 'value': 0.0, 'option4': 1.0, 'option2': 1.0, 'option3': 1.0, 'option0': 1.0, 'option1': 1.0},
             'f_step_offset ': {'use': 1.0, 'min': 0.0, 'max': 1.0, 'value': 0.0, 'option4': 1.0, 'option2': 1.0, 'option3': 1.0, 'option0': 1.0, 'option1': 1.0},
             'f_step_quadratic ': {'use': 1.0, 'min': 0.0, 'max': 0.0, 'value': 0.0, 'option4': 1.0, 'option2': 1.0, 'option3': 1.0, 'option0': 1.0, 'option1': 1.0},
             'f_tail_linear ': {'use': 1.0, 'min': 0.0, 'max': 1.0, 'value': 0.01, 'option4': 1.0, 'option2': 1.0, 'option3': 2.0, 'option0': 1.0, 'option1': 1.0},
             'f_tail_offset ': {'use': 1.0, 'min': 0.0, 'max': 0.1, 'value': 0.04, 'option4': 1.0, 'option2': 1.0, 'option3': 2.0, 'option0': 1.0, 'option1': 1.0},
             'f_tail_quadratic ': {'use': 1.0, 'min': 0.0, 'max': 0.01, 'value': 0.0, 'option4': 1.0, 'option2': 1.0, 'option3': 1.0, 'option0': 1.0, 'option1': 1.0},
             'fwhm_fanoprime ': {'use': 2.0, 'min': 1e-06, 'max': 0.05, 'value': 0.00012, 'option4': 1.0, 'option2': 2.0, 'option3': 2.0, 'option0': 1.0, 'option1': 1.0},
             'fwhm_offset ': {'use': 2.0, 'min': 0.005, 'max': 0.5, 'value': 0.12, 'option4': 1.0, 'option2': 2.0, 'option3': 2.0, 'option0': 1.0, 'option1': 1.0},
             'gamma_linear ': {'use': 1.0, 'min': 0.0, 'max': 3.0, 'value': 0.0, 'option4': 1.0, 'option2': 1.0, 'option3': 1.0, 'option0': 1.0, 'option1': 1.0},
             'gamma_offset ': {'use': 1.0, 'min': 0.1, 'max': 10.0, 'value': 2.0, 'option4': 1.0, 'option2': 1.0, 'option3': 1.0, 'option0': 1.0, 'option1': 1.0},
             'gamma_quadratic ': {'use': 1.0, 'min': 0.0, 'max': 0.0, 'value': 0.0, 'option4': 1.0, 'option2': 1.0, 'option3': 1.0, 'option0': 1.0, 'option1': 1.0},
             'ge_escape ': {'use': 1.0, 'min': 0.0, 'max': 1.0, 'value': 0.0, 'option4': 1.0, 'option2': 1.0, 'option3': 1.0, 'option0': 1.0, 'option1': 1.0},
             'kb_f_tail_linear ': {'use': 1.0, 'min': 0.0, 'max': 0.02, 'value': 0.0, 'option4': 1.0, 'option2': 1.0, 'option3': 2.0, 'option0': 1.0, 'option1': 1.0},
             'kb_f_tail_offset ': {'use': 1.0, 'min': 0.0, 'max': 0.2, 'value': 0.0, 'option4': 1.0, 'option2': 1.0, 'option3': 2.0, 'option0': 1.0, 'option1': 1.0},
             'kb_f_tail_quadratic ': {'use': 1.0, 'min': 0.0, 'max': 0.0, 'value': 0.0, 'option4': 1.0, 'option2': 1.0, 'option3': 1.0, 'option0': 1.0, 'option1': 1.0},
             'linear ': {'use': 1.0, 'min': 0.0, 'max': 1.0, 'value': 0.0, 'option4': 1.0, 'option2': 1.0, 'option3': 1.0, 'option0': 1.0, 'option1': 1.0},
             'pileup0 ': {'use': 1.0, 'min': -10.0, 'max': 1.0, 'value': 1e-10, 'option4': 1.0, 'option2': 1.0, 'option3': 1.0, 'option0': 1.0, 'option1': 1.0},
             'pileup1 ': {'use': 1.0, 'min': -10.0, 'max': 1.0, 'value': 1e-10, 'option4': 1.0, 'option2': 1.0, 'option3': 1.0, 'option0': 1.0, 'option1': 1.0},
             'pileup2 ': {'use': 1.0, 'min': -10.0, 'max': 1.0, 'value': 1e-10, 'option4': 1.0, 'option2': 1.0, 'option3': 1.0, 'option0': 1.0, 'option1': 1.0},
             'pileup3 ': {'use': 1.0, 'min': -10.0, 'max': 1.0, 'value': 1e-10, 'option4': 1.0, 'option2': 1.0, 'option3': 1.0, 'option0': 1.0, 'option1': 1.0},
             'pileup4 ': {'use': 1.0, 'min': -10.0, 'max': 1.0, 'value': 1e-10, 'option4': 1.0, 'option2': 1.0, 'option3': 1.0, 'option0': 1.0, 'option1': 1.0},
             'pileup5 ': {'use': 1.0, 'min': -10.0, 'max': 1.0, 'value': 1e-10, 'option4': 1.0, 'option2': 1.0, 'option3': 1.0, 'option0': 1.0, 'option1': 1.0},
             'pileup6 ': {'use': 1.0, 'min': -10.0, 'max': 1.0, 'value': 1e-10, 'option4': 1.0, 'option2': 1.0, 'option3': 1.0, 'option0': 1.0, 'option1': 1.0},
             'pileup7 ': {'use': 1.0, 'min': -10.0, 'max': 1.0, 'value': 1e-10, 'option4': 1.0, 'option2': 1.0, 'option3': 1.0, 'option0': 1.0, 'option1': 1.0},
             'pileup8 ': {'use': 1.0, 'min': -10.0, 'max': 1.0, 'value': 1e-10, 'option4': 1.0, 'option2': 1.0, 'option3': 1.0, 'option0': 1.0, 'option1': 1.0},
             'si_escape ': {'use': 1.0, 'min': 0.0, 'max': 0.5, 'value': 0.0, 'option4': 1.0, 'option2': 1.0, 'option3': 1.0, 'option0': 1.0, 'option1': 1.0},
             'snip_width ': {'use': 1.0, 'min': 0.1, 'max': 2.82842712475, 'value': 0.15, 'option4': 1.0, 'option2': 1.0, 'option3': 1.0, 'option0': 1.0, 'option1': 1.0},
             }
