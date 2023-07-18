# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# @author: Li Li (lili@bnl.gov)                                        #
# created on 08/19/2014                                                #
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
import six
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal, assert_raises

from skbeam.core.constants.basic import basic
from skbeam.core.constants.xrf import XrayLibWrap, XrayLibWrap_Energy, XrfElement, emission_line_search
from skbeam.core.utils import NotInstalledError


def test_element_data():
    """
    smoke test of all elements
    """

    data1 = []
    data2 = []

    name_list = []
    for i in range(100):
        e = XrfElement(i + 1)
        data1.append(e.cs(10)["Ka1"])
        name_list.append(e.name)

    for item in name_list:
        e = XrfElement(item)
        data2.append(e.cs(10)["Ka1"])

    assert_array_equal(data1, data2)

    return


def test_element_finder():
    true_name = sorted(["Eu", "Cu"])
    out = emission_line_search(8, 0.05, 10)
    found_name = sorted(list(six.iterkeys(out)))
    assert_equal(true_name, found_name)
    return


def test_XrayLibWrap_notpresent():
    from skbeam.core.constants import xrf

    # stash the original xraylib object
    xraylib = xrf.xraylib
    # force the not present exception to be raised by setting xraylib to None
    xrf.xraylib = None
    assert_raises(NotInstalledError, xrf.XrfElement, None)
    assert_raises(NotInstalledError, xrf.emission_line_search, None, None, None)
    assert_raises(NotInstalledError, xrf.XrayLibWrap, None, None)
    assert_raises(NotInstalledError, xrf.XrayLibWrap_Energy, None, None, None)
    # reset xraylib so nothing else breaks
    xrf.xraylib = xraylib


def test_XrayLibWrap():
    for Z in range(1, 101):
        for infotype in XrayLibWrap.opts_info_type:
            xlw = XrayLibWrap(Z, infotype)
            assert xlw.all is not None
            for key in xlw:
                assert xlw[key] is not None
            assert_equal(xlw.info_type, infotype)
            # make sure len doesn't break
            len(xlw)


def test_XrayLibWrap_Energy():
    for Z in range(1, 101):
        for infotype in XrayLibWrap_Energy.opts_info_type:
            incident_energy = 10
            xlwe = XrayLibWrap_Energy(element=Z, info_type=infotype, incident_energy=incident_energy)
            incident_energy *= 2
            xlwe.incident_energy = incident_energy
            assert_equal(xlwe.incident_energy, incident_energy)
            assert_equal(xlwe.info_type, infotype)


def test_cs_different_units():
    e = XrfElement("Fe")
    # test at different energies
    for eng in range(10, 20):
        cs1 = np.array([v for k, v in e.cs(eng).all])  # unit in cm2/g
        cs2 = np.array([v for k, v in e.csb(eng).all])  # unit in barns/atom
        cs1 /= cs1[0]
        cs2 /= cs2[0]
        # ratio should be the same no matter which unit is used
        assert_array_almost_equal(cs1, cs2, decimal=10)


def test_element_creation():
    prev_element = None
    elements = [elm for abbrev, elm in six.iteritems(basic) if isinstance(abbrev, int)]
    elements.sort()
    for element in elements:
        Z = element.Z
        element.mass
        element.density
        sym = element.sym
        inits = [Z, sym, sym.upper(), sym.lower(), sym.swapcase()]
        element = None
        for init in inits:
            element = XrfElement(init)
            # obtain the next four attributes to make sure the XrayLibWrap is
            # working
            element.bind_energy
            element.fluor_yield
            element.jump_factor
            element.emission_line.all
            if prev_element is not None:
                # compare prev_element to element
                assert_equal(prev_element.__lt__(element), True)
                assert_equal(prev_element < element, True)
                assert_equal(prev_element.__eq__(element), False)
                assert_equal(prev_element == element, False)
                assert_equal(prev_element >= element, False)
                assert_equal(prev_element > element, False)
                # compare element to prev_element
                assert_equal(element < prev_element, False)
                assert_equal(element.__lt__(prev_element), False)
                assert_equal(element <= prev_element, False)
                assert_equal(element.__eq__(prev_element), False)
                assert_equal(element == prev_element, False)
                assert_equal(element >= prev_element, True)
                assert_equal(element > prev_element, True)
        prev_element = element
