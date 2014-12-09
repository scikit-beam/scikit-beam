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

"""
    This module is for test output.py saving integrated powder
    x-ray diffraction intensities into  different file formats.
    (Output into different file formats, .chi, .dat, .xye, .gsas)

"""

from __future__ import (absolute_import, division,
                       unicode_literals, print_function)
import six
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_equal, assert_not_equal
import six
from nose.tools import raises

from skxray.testing.decorators import known_fail_if
import skxray.io.output as output


def test_save_output():
    filename = "sinvalues"
    x = np.linspace(0, 100)
    y = np.sin(x)

    output.save_output(x,y, filename, q_or_2theta="Q")
    output.save_output(x, y, filename, q_or_2theta="2theta", ext=".dat", err=None, dir_path=None)

    Data_chi = np.loadtxt("sinvalues.chi", skiprows=9)
    Data_dat = np.loadtxt("sinvalues.dat", skiprows=9)

    assert_array_almost_equal(x, Data_chi[:, 0])
    assert_array_almost_equal(y, Data_chi[:, 1])

    assert_array_almost_equal(x, Data_dat[:, 0])
    assert_array_almost_equal(y, Data_dat[:, 1])

    return


def test_save_gsas():
    pass