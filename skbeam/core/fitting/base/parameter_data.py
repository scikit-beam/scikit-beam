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
from __future__ import absolute_import, division, print_function

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

Different fitting strategies are included to turn on or turn off some
parameters. Those strategies are default, linear, free_energy, free_all and
e_calibration. They are empirical experience from authors of the original code.
"""


# old param dict, keep it here for now.
para_dict = {
    "coherent_sct_amplitude": {"bound_type": "none", "min": 7.0, "max": 8.0, "value": 6.0},
    "coherent_sct_energy": {"bound_type": "none", "min": 10.4, "max": 12.4, "value": 11.8},
    "compton_amplitude": {"bound_type": "none", "min": 0.0, "max": 10.0, "value": 5.0},
    "compton_angle": {"bound_type": "lohi", "min": 75.0, "max": 90.0, "value": 90.0},
    "compton_f_step": {"bound_type": "lohi", "min": 0.0, "max": 1.5, "value": 0.1},
    "compton_f_tail": {"bound_type": "lohi", "min": 0.0, "max": 3.0, "value": 0.8},
    "compton_fwhm_corr": {"bound_type": "lohi", "min": 0.1, "max": 3.0, "value": 1.4},
    "compton_gamma": {"bound_type": "none", "min": 0.1, "max": 10.0, "value": 1.0},
    "compton_hi_f_tail": {"bound_type": "none", "min": 1e-06, "max": 1.0, "value": 0.01},
    "compton_hi_gamma": {"bound_type": "none", "min": 0.1, "max": 3.0, "value": 1.0},
    "e_linear": {"bound_type": "fixed", "min": 0.001, "max": 0.1, "value": 1.0},
    "e_offset": {"bound_type": "fixed", "min": -0.2, "max": 0.2, "value": 0.0},
    "e_quadratic": {"bound_type": "none", "min": -0.0001, "max": 0.0001, "value": 0.0},
    "f_step_linear": {"bound_type": "none", "min": 0.0, "max": 1.0, "value": 0.0},
    "f_step_offset": {"bound_type": "none", "min": 0.0, "max": 1.0, "value": 0.0},
    "f_step_quadratic": {"bound_type": "none", "min": 0.0, "max": 0.0, "value": 0.0},
    "f_tail_linear": {"bound_type": "none", "min": 0.0, "max": 1.0, "value": 0.01},
    "f_tail_offset": {"bound_type": "none", "min": 0.0, "max": 0.1, "value": 0.04},
    "f_tail_quadratic": {"bound_type": "none", "min": 0.0, "max": 0.01, "value": 0.0},
    "fwhm_fanoprime": {"bound_type": "lohi", "min": 1e-06, "max": 0.05, "value": 0.00012},
    "fwhm_offset": {"bound_type": "lohi", "min": 0.005, "max": 0.5, "value": 0.12},
    "gamma_linear": {"bound_type": "none", "min": 0.0, "max": 3.0, "value": 0.0},
    "gamma_offset": {"bound_type": "none", "min": 0.1, "max": 10.0, "value": 2.0},
    "gamma_quadratic": {"bound_type": "none", "min": 0.0, "max": 0.0, "value": 0.0},
    "ge_escape": {"bound_type": "none", "min": 0.0, "max": 1.0, "value": 0.0},
    "kb_f_tail_linear": {"bound_type": "none", "min": 0.0, "max": 0.02, "value": 0.0},
    "kb_f_tail_offset": {"bound_type": "none", "min": 0.0, "max": 0.2, "value": 0.0},
    "kb_f_tail_quadratic": {"bound_type": "none", "min": 0.0, "max": 0.0, "value": 0.0},
    "linear": {"bound_type": "none", "min": 0.0, "max": 1.0, "value": 0.0},
    "pileup0": {"bound_type": "none", "min": -10.0, "max": 1.10, "value": 1e-10},
    "pileup1": {"bound_type": "none", "min": -10.0, "max": 1.10, "value": 1e-10},
    "pileup2": {"bound_type": "none", "min": -10.0, "max": 1.10, "value": 1e-10},
    "pileup3": {"bound_type": "none", "min": -10.0, "max": 1.10, "value": 1e-10},
    "pileup4": {"bound_type": "none", "min": -10.0, "max": 1.10, "value": 1e-10},
    "pileup5": {"bound_type": "none", "min": -10.0, "max": 1.10, "value": 1e-10},
    "pileup6": {"bound_type": "none", "min": -10.0, "max": 1.10, "value": 1e-10},
    "pileup7": {"bound_type": "none", "min": -10.0, "max": 1.10, "value": 1e-10},
    "pileup8": {"bound_type": "none", "min": -10.0, "max": 1.10, "value": 1e-10},
    "si_escape": {"bound_type": "none", "min": 0.0, "max": 0.5, "value": 0.0},
    "snip_width": {"bound_type": "none", "min": 0.1, "max": 2.82842712475, "value": 0.15},
}


# fitting strategies
adjust_element = {
    "coherent_sct_amplitude": "none",
    "coherent_sct_energy": "fixed",
    "compton_amplitude": "none",
    "compton_angle": "fixed",
    "compton_f_step": "fixed",
    "compton_f_tail": "fixed",
    "compton_fwhm_corr": "lohi",
    "compton_gamma": "fixed",
    "compton_hi_f_tail": "fixed",
    "compton_hi_gamma": "fixed",
    "e_linear": "fixed",
    "e_offset": "fixed",
    "e_quadratic": "fixed",
    "fwhm_fanoprime": "fixed",
    "fwhm_offset": "fixed",
    "non_fitting_values": "fixed",
}

e_calibration = {
    "coherent_sct_amplitude": "none",
    "coherent_sct_energy": "fixed",
    "compton_amplitude": "none",
    "compton_angle": "fixed",
    "compton_f_step": "fixed",
    "compton_f_tail": "fixed",
    "compton_fwhm_corr": "fixed",
    "compton_gamma": "fixed",
    "compton_hi_f_tail": "fixed",
    "compton_hi_gamma": "fixed",
    "e_linear": "lohi",
    "e_offset": "lohi",
    "e_quadratic": "fixed",
    "fwhm_fanoprime": "fixed",
    "fwhm_offset": "fixed",
    "non_fitting_values": "fixed",
}

linear = {
    "coherent_sct_amplitude": "none",
    "coherent_sct_energy": "fixed",
    "compton_amplitude": "none",
    "compton_angle": "fixed",
    "compton_f_step": "fixed",
    "compton_f_tail": "fixed",
    "compton_fwhm_corr": "fixed",
    "compton_gamma": "fixed",
    "compton_hi_f_tail": "fixed",
    "compton_hi_gamma": "fixed",
    "e_linear": "fixed",
    "e_offset": "fixed",
    "e_quadratic": "fixed",
    "fwhm_fanoprime": "fixed",
    "fwhm_offset": "fixed",
    "non_fitting_values": "fixed",
}

free_more = {
    "coherent_sct_amplitude": "none",
    "coherent_sct_energy": "lohi",
    "compton_amplitude": "none",
    "compton_angle": "lohi",
    "compton_f_step": "lohi",
    "compton_f_tail": "fixed",
    "compton_fwhm_corr": "lohi",
    "compton_gamma": "lohi",
    "compton_hi_f_tail": "fixed",
    "compton_hi_gamma": "fixed",
    "e_linear": "lohi",
    "e_offset": "lohi",
    "e_quadratic": "lohi",
    "fwhm_fanoprime": "lohi",
    "fwhm_offset": "lohi",
    "non_fitting_values": "fixed",
}

fit_with_tail = {
    "coherent_sct_amplitude": "none",
    "coherent_sct_energy": "lohi",
    "compton_amplitude": "none",
    "compton_angle": "lohi",
    "compton_f_step": "fixed",
    "compton_f_tail": "lohi",
    "compton_fwhm_corr": "lohi",
    "compton_gamma": "fixed",
    "compton_hi_f_tail": "fixed",
    "compton_hi_gamma": "fixed",
    "e_linear": "lohi",
    "e_offset": "lohi",
    "e_quadratic": "lohi",
    "fwhm_fanoprime": "lohi",
    "fwhm_offset": "lohi",
    "non_fitting_values": "fixed",
}


default_param = {
    "coherent_sct_amplitude": {"bound_type": "none", "max": 10000000.0, "min": 0.10, "value": 100000},
    "coherent_sct_energy": {
        "bound_type": "lohi",
        "description": "Incident E [keV]",
        "max": 13.0,
        "min": 9.0,
        "value": 10.0,
    },
    "compton_amplitude": {"bound_type": "none", "max": 10000000.0, "min": 0.10, "value": 100000.0},
    "compton_angle": {"bound_type": "lohi", "max": 100.0, "min": 80.0, "value": 90.0},
    "compton_f_step": {"bound_type": "fixed", "max": 0.01, "min": 0.0, "value": 0.01},
    "compton_f_tail": {"bound_type": "fixed", "max": 0.3, "min": 0.0001, "value": 0.05},
    "compton_fwhm_corr": {
        "bound_type": "lohi",
        "description": "fwhm Coef, Compton",
        "max": 2.5,
        "min": 0.5,
        "value": 1.5,
    },
    "compton_gamma": {"bound_type": "lohi", "max": 4.2, "min": 3.8, "value": 4.0},
    "compton_hi_f_tail": {"bound_type": "fixed", "max": 1.0, "min": 1e-06, "value": 0.1},
    "compton_hi_gamma": {"bound_type": "fixed", "max": 3.0, "min": 0.1, "value": 2.0},
    "e_linear": {
        "bound_type": "lohi",
        "description": "E Calib. Coef, a1",
        "max": 0.011,
        "min": 0.009,
        "tool_tip": "E(channel) = a0 + a1*channel+ a2*channel**2",
        "value": 0.01,
    },
    "e_offset": {
        "bound_type": "lohi",
        "description": "E Calib. Coef, a0",
        "max": 0.015,
        "min": -0.01,
        "tool_tip": "E(channel) = a0 + a1*channel+ a2*channel**2",
        "value": 0.0,
    },
    "e_quadratic": {
        "bound_type": "lohi",
        "description": "E Calib. Coef, a2",
        "max": 1e-06,
        "min": -1e-06,
        "tool_tip": "E(channel) = a0 + a1*channel+ a2*channel**2",
        "value": 0.0,
    },
    "fwhm_fanoprime": {
        "bound_type": "fixed",
        "description": "fwhm Coef, b2",
        "max": 0.0001,
        "min": 1e-07,
        "value": 1e-06,
    },
    "fwhm_offset": {
        "bound_type": "lohi",
        "description": "fwhm Coef, b1 [keV]",
        "max": 0.19,
        "min": 0.16,
        "tool_tip": "width**2 = (b1/2.3548)**2 + 3.85*b2*E",
        "value": 0.178,
    },
    "non_fitting_values": {
        "element_list": ["Ar", "Fe", "Ce_L", "Pt_M"],
        "energy_bound_low": {"value": 1.5, "default_value": 1.5, "description": "E low [keV]"},
        "energy_bound_high": {"value": 13.5, "default_value": 13.5, "description": "E high [keV]"},
        "epsilon": 3.51,  # electron hole energy
        "background_width": 0.5,
    },
}


def get_para():
    """More to be added here.
    The para_dict will be updated
    based on different algorithms.
    Use copy for dict.
    """
    return default_param
