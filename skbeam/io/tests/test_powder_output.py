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
    (Output into different file formats, .chi, .dat, .xye, gsas)
    Added a test to check the GSAS file reader and file writer
"""
from __future__ import absolute_import, division, print_function

import math
import os

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

import skbeam.io.save_powder_output as output
from skbeam.io.gsas_file_reader import gsas_reader
from skbeam.io.save_powder_output import gsas_writer


def test_save_output():
    filename = "function_values"
    x = np.arange(0, 100, 1)
    y = np.exp(x)
    y1 = y * math.erf(0.5)

    output.save_output(x, y, filename, q_or_2theta="Q", err=None, dir_path=None)
    output.save_output(x, y, filename, q_or_2theta="2theta", ext=".dat", err=None, dir_path=None)
    output.save_output(x, y, filename, q_or_2theta="2theta", ext=".xye", err=y1, dir_path=None)

    Data_chi = np.loadtxt("function_values.chi", skiprows=7)
    Data_dat = np.loadtxt("function_values.dat", skiprows=7)
    Data_xye = np.loadtxt("function_values.xye", skiprows=7)

    assert_array_almost_equal(x, Data_chi[:, 0])
    assert_array_almost_equal(y, Data_chi[:, 1])

    assert_array_almost_equal(x, Data_dat[:, 0])
    assert_array_almost_equal(y, Data_dat[:, 1])

    assert_array_almost_equal(x, Data_xye[:, 0])
    assert_array_almost_equal(y, Data_xye[:, 1])
    assert_array_almost_equal(y1, Data_xye[:, 2])

    os.remove("function_values.chi")
    os.remove("function_values.dat")
    os.remove("function_values.xye")


@pytest.mark.skipif(os.name == "nt", reason="The tested functions are not working on Windows.")
def test_gsas_output(tmpdir):
    filename = os.path.join(tmpdir, "function_values")
    x = np.arange(0, 100, 5)
    y = np.arange(0, 200, 10)
    err = y * math.erf(0.2)

    vi = []
    esd_vi = []
    for ei in err:
        if ei > 0.0:
            vi.append(1.0 / ei**2)
            esd_vi.append(1.0 / round(ei) ** 2)
        else:
            vi.append(0.0)
            esd_vi.append(0.0)

    gsas_writer(x, y, filename + "_std", mode=None, err=None, dir_path=None)
    gsas_writer(x, y, filename + "_esd", mode="ESD", err=err, dir_path=None)
    gsas_writer(x, y, filename + "_fxye", mode="FXYE", err=err, dir_path=None)

    tth1, intensity1, err1 = gsas_reader(filename + "_std.gsas")
    tth2, intensity2, err2 = gsas_reader(filename + "_esd.gsas")
    tth3, intensity3, err3 = gsas_reader(filename + "_fxye.gsas")

    assert_array_equal(x, tth1)
    assert_array_equal(x, tth2)
    assert_array_equal(x, tth3)

    assert_array_equal(y, intensity1)
    assert_array_equal(y, intensity2)
    assert_array_equal(y, intensity3)

    assert_array_equal(esd_vi, err2)
    assert_array_almost_equal(vi, err3, decimal=12)
