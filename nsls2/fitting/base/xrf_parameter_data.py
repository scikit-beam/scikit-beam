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

option0, option1, ..., option4:
Those are different strategies to turn on or turn off some parameters.
They are empirical experience from authors of the original code.
"""


para_dict = {'coherent_sct_amplitude ': {'use': 'none', 'min': 7.0, 'max': 8.0, 'value': 6.0, 'option4': 'none', 'option2': 'none', 'option3': 'none', 'option0': 'none', 'option1': 'none'},
             'coherent_sct_energy ': {'use': 'none', 'min': 10.4, 'max': 12.4, 'value': 11.8, 'option4': 'fixed', 'option2': 'lohi', 'option3': 'lohi', 'option0': 'fixed', 'option1': 'fixed'},
             'compton_amplitude ': {'use': 'none', 'min': 0.0, 'max': 10.0, 'value': 5.0, 'option4': 'none', 'option2': 'none', 'option3': 'none', 'option0': 'none', 'option1': 'none'},
             'compton_angle ': {'use': 'lohi', 'min': 75.0, 'max': 90.0, 'value': 90.0, 'option4': 'fixed', 'option2': 'lohi', 'option3': 'lohi', 'option0': 'fixed', 'option1': 'fixed'},
             'compton_f_step ': {'use': 'lohi', 'min': 0.0, 'max': 1.5, 'value': 0.1, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'lohi', 'option0': 'fixed', 'option1': 'fixed'},
             'compton_f_tail ': {'use': 'lohi', 'min': 0.0, 'max': 3.0, 'value': 0.8, 'option4': 'fixed', 'option2': 'lo', 'option3': 'fixed', 'option0': 'fixed', 'option1': 'fixed'},
             'compton_fwhm_corr ': {'use': 'lohi', 'min': 0.1, 'max': 3.0, 'value': 1.4, 'option4': 'fixed', 'option2': 'lohi', 'option3': 'lohi', 'option0': 'fixed', 'option1': 'fixed'},
             'compton_gamma ': {'use': 'none', 'min': 0.1, 'max': 10.0, 'value': 1.0, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'lohi', 'option0': 'fixed', 'option1': 'fixed'},
             'compton_hi_f_tail ': {'use': 'none', 'min': 1e-06, 'max': 1.0, 'value': 0.01, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'fixed', 'option0': 'fixed', 'option1': 'fixed'},
             'compton_hi_gamma ': {'use': 'none', 'min': 0.1, 'max': 3.0, 'value': 1.0, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'fixed', 'option0': 'fixed', 'option1': 'fixed'},
             'e_linear ': {'use': 'fixed', 'min': 0.001, 'max': 0.1, 'value': 1.0, 'option4': 'lohi', 'option2': 'lohi', 'option3': 'lohi', 'option0': 'fixed', 'option1': 'fixed'},
             'e_offset ': {'use': 'fixed', 'min': -0.2, 'max': 0.2, 'value': 0.0, 'option4': 'lohi', 'option2': 'lohi', 'option3': 'lohi', 'option0': 'fixed', 'option1': 'fixed'},
             'e_quadratic ': {'use': 'none', 'min': -0.0001, 'max': 0.0001, 'value': 0.0, 'option4': 'lohi', 'option2': 'lohi', 'option3': 'lohi', 'option0': 'fixed', 'option1': 'fixed'},
             'f_step_linear ': {'use': 'none', 'min': 0.0, 'max': 1.0, 'value': 0.0, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'fixed', 'option0': 'fixed', 'option1': 'fixed'},
             'f_step_offset ': {'use': 'none', 'min': 0.0, 'max': 1.0, 'value': 0.0, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'fixed', 'option0': 'fixed', 'option1': 'fixed'},
             'f_step_quadratic ': {'use': 'none', 'min': 0.0, 'max': 0.0, 'value': 0.0, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'fixed', 'option0': 'fixed', 'option1': 'fixed'},
             'f_tail_linear ': {'use': 'none', 'min': 0.0, 'max': 1.0, 'value': 0.01, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'lohi', 'option0': 'fixed', 'option1': 'fixed'},
             'f_tail_offset ': {'use': 'none', 'min': 0.0, 'max': 0.1, 'value': 0.04, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'lohi', 'option0': 'fixed', 'option1': 'fixed'},
             'f_tail_quadratic ': {'use': 'none', 'min': 0.0, 'max': 0.01, 'value': 0.0, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'fixed', 'option0': 'fixed', 'option1': 'fixed'},
             'fwhm_fanoprime ': {'use': 'lohi', 'min': 1e-06, 'max': 0.05, 'value': 0.00012, 'option4': 'fixed', 'option2': 'lohi', 'option3': 'lohi', 'option0': 'fixed', 'option1': 'fixed'},
             'fwhm_offset ': {'use': 'lohi', 'min': 0.005, 'max': 0.5, 'value': 0.12, 'option4': 'fixed', 'option2': 'lohi', 'option3': 'lohi', 'option0': 'fixed', 'option1': 'fixed'},
             'gamma_linear ': {'use': 'none', 'min': 0.0, 'max': 3.0, 'value': 0.0, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'fixed', 'option0': 'fixed', 'option1': 'fixed'},
             'gamma_offset ': {'use': 'none', 'min': 0.1, 'max': 10.0, 'value': 2.0, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'fixed', 'option0': 'fixed', 'option1': 'fixed'},
             'gamma_quadratic ': {'use': 'none', 'min': 0.0, 'max': 0.0, 'value': 0.0, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'fixed', 'option0': 'fixed', 'option1': 'fixed'},
             'ge_escape ': {'use': 'none', 'min': 0.0, 'max': 1.0, 'value': 0.0, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'fixed', 'option0': 'fixed', 'option1': 'fixed'},
             'kb_f_tail_linear ': {'use': 'none', 'min': 0.0, 'max': 0.02, 'value': 0.0, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'lohi', 'option0': 'fixed', 'option1': 'fixed'},
             'kb_f_tail_offset ': {'use': 'none', 'min': 0.0, 'max': 0.2, 'value': 0.0, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'lohi', 'option0': 'fixed', 'option1': 'fixed'},
             'kb_f_tail_quadratic ': {'use': 'none', 'min': 0.0, 'max': 0.0, 'value': 0.0, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'fixed', 'option0': 'fixed', 'option1': 'fixed'},
             'linear ': {'use': 'none', 'min': 0.0, 'max': 1.0, 'value': 0.0, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'fixed', 'option0': 'fixed', 'option1': 'fixed'},
             'pileup0 ': {'use': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'fixed', 'option0': 'fixed', 'option1': 'fixed'},
             'pileup1 ': {'use': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'fixed', 'option0': 'fixed', 'option1': 'fixed'},
             'pileup2 ': {'use': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'fixed', 'option0': 'fixed', 'option1': 'fixed'},
             'pileup3 ': {'use': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'fixed', 'option0': 'fixed', 'option1': 'fixed'},
             'pileup4 ': {'use': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'fixed', 'option0': 'fixed', 'option1': 'fixed'},
             'pileup5 ': {'use': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'fixed', 'option0': 'fixed', 'option1': 'fixed'},
             'pileup6 ': {'use': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'fixed', 'option0': 'fixed', 'option1': 'fixed'},
             'pileup7 ': {'use': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'fixed', 'option0': 'fixed', 'option1': 'fixed'},
             'pileup8 ': {'use': 'none', 'min': -10.0, 'max': 1.10, 'value': 1e-10, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'fixed', 'option0': 'fixed', 'option1': 'fixed'},
             'si_escape ': {'use': 'none', 'min': 0.0, 'max': 0.5, 'value': 0.0, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'fixed', 'option0': 'fixed', 'option1': 'fixed'},
             'snip_width ': {'use': 'none', 'min': 0.1, 'max': 2.82842712475, 'value': 0.15, 'option4': 'fixed', 'option2': 'fixed', 'option3': 'fixed', 'option0': 'fixed', 'option1': 'fixed'},
             }
