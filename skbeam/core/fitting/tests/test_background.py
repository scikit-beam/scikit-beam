# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# @author: Li Li (lili@bnl.gov)                                        #
# created on 08/16/2014                                                #
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

import numpy as np
from numpy.testing import assert_allclose

from skbeam.core.fitting import snip_method


def test_snip_method():
    """
    test of background function from xrf fit
    """

    xmin = 0

    # three gaussian peak
    xval = np.arange(-20, 20, 0.1)
    std = 0.01
    yval1 = np.exp(-(xval**2) / 2 / std**2)
    yval2 = np.exp(-((xval - 10) ** 2) / 2 / std**2)
    yval3 = np.exp(-((xval + 10) ** 2) / 2 / std**2)

    # background as exponential
    a0 = 1.0
    a1 = 0.1
    a2 = 0.5
    bg_true = a0 * np.exp(-xval * a1 + a2)

    yval = yval1 + yval2 + yval3 + bg_true

    bg = snip_method(yval, 0.0, 1.0, 0.0, xmin=xmin, xmax=3000, spectral_binning=None, width=0.1)

    # ignore the boundary part
    cutval = 15
    bg_true_part = bg_true[cutval:-cutval]
    bg_cal_part = bg[cutval:-cutval]

    assert_allclose(bg_true_part, bg_cal_part, rtol=1e-3, atol=1e-1)
