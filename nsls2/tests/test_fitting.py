# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
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

import numpy as np
import six
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)

from nsls2.testing.decorators import known_fail_if
from nsls2.fitting.model.physics_model import (gauss_fit, lorentzian_fit,
                                               lorentzian2_fit)

@known_fail_if(True)
def test_fit_quad_to_peak():
    assert(False)


def test_gauss_fit():
    x = np.arange(-1, 1, 0.01)
    area = 1
    cen = 0
    std = 1
    true_val = [area, cen, std]
    y = area / np.sqrt(2 * np.pi) / std * np.exp(-(x - cen)**2 / 2 / std**2)

    out = gauss_fit([x, y],
                    0.8, 'free', [0, 1],
                    0.1, 'free', [0, 0.5],
                    0.5, 'free', [0, 1])

    fitted_val = (out[0]['area'], out[0]['center'], out[0]['sigma'])
    assert_array_almost_equal(true_val, fitted_val)

    return


def test_lorentzian_fit():
    x = np.arange(-1, 1, 0.01)
    area = 1
    center = 0
    sigma = 1
    true_val = [area, center, sigma]

    y = (area/(1 + ((x - center) / sigma)**2)) / (np.pi * sigma)

    out = lorentzian_fit([x, y],
                         0.8, 'free', [0, 1],
                         0.1, 'free', [0, 0.5],
                         0.5, 'free', [0, 1])

    fitted_val = (out[0]['area'], out[0]['center'], out[0]['sigma'])
    assert_array_almost_equal(true_val, fitted_val)

    return


def test_lorentzian2_fit():
    x = np.arange(-1, 1, 0.01)
    area = 1
    center = 0
    sigma = 1
    true_val = [area, center, sigma]

    y = (area/(1 + ((x - center) / sigma)**2)**2) / (np.pi * sigma)

    out = lorentzian2_fit([x, y],
                          0.8, 'free', [0, 1],
                          0.1, 'free', [0, 0.5],
                          0.5, 'free', [0, 1])
    fitted_val = (out[0]['area'], out[0]['center'], out[0]['sigma'])
    assert_array_almost_equal(true_val, fitted_val)

    return
