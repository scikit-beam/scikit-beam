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


from __future__ import (absolute_import, division,
                        unicode_literals, print_function)
import six
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_equal, assert_not_equal

from skxray.constants.api import (XrfElement, emission_line_search,
                                  calibration_standards, HKL)

from skxray.core import (q_to_d, d_to_q)

def test_element_data():
    """
    smoke test of all elements
    """

    data1 = []
    data2 = []

    name_list = []
    for i in range(100):
        e = XrfElement(i+1)
        data1.append(e.cs(10)['Ka1'])
        name_list.append(e.name)

    for item in name_list:
        e = XrfElement(item)
        data2.append(e.cs(10)['Ka1'])

    assert_array_equal(data1, data2)

    return


def test_element_finder():

    true_name = sorted(['Eu', 'Cu'])
    out = emission_line_search(8, 0.05, 10)
    found_name = sorted(list(six.iterkeys(out)))
    assert_equal(true_name, found_name)
    return


def test_XrayLibWrap():
    from skxray.constants.xrf import XrayLibWrap, XrayLibWrap_Energy
    for Z in range(1, 101):
        for infotype in XrayLibWrap.opts_info_type:
            xlw = XrayLibWrap(Z, infotype)
            assert_not_equal(xlw.all, None)
            for key in xlw:
                assert_not_equal(xlw[key], None)
            assert_equal(xlw.info_type, infotype)
            # make sure len doesn't break
            len(xlw)
        for infotype in XrayLibWrap_Energy.opts_info_type:
            incident_energy = 10
            xlwe = XrayLibWrap_Energy(element=Z,
                                      info_type=infotype,
                                      incident_energy=incident_energy)
            incident_energy *= 2
            xlwe.incident_energy = incident_energy
            assert_equal(xlwe.incident_energy, incident_energy)
            assert_equal(xlwe.info_type, infotype)


def smoke_test_element_creation():
    from skxray.constants.basic import basic
    prev_element = None
    elements = [elm for abbrev, elm in six.iteritems(basic)
                if isinstance(abbrev, int)]
    elements.sort()
    for element in elements:
        Z = element.Z
        mass = element.mass
        density = element.density
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
            assert_equal(element.Z, Z)
            assert_equal(element.mass, mass)
            desc = six.text_type(element)
            assert_equal(desc, "Element name " + six.text_type(sym) +
                         " with atomic Z " + six.text_type(Z))
            if not np.isnan(density):
                # shield the assertion from any elements whose density is
                # unknown
                assert_equal(element.density, density)
            assert_equal(element.name, sym)
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
                # create a second instance of element with the same Z value and test its comparison
                element_2 = XrfElement(element.Z)
                assert_equal(element < element_2, False)
                assert_equal(element <= element_2, True)
                assert_equal(element == element_2, True)
                assert_equal(element >= element_2, True)
                assert_equal(element_2 > element, False)
                assert_equal(element_2 < element, False)
                assert_equal(element_2 <= element, True)
                assert_equal(element_2 == element, True)
                assert_equal(element_2 >= element, True)
                assert_equal(element_2 > element, False)
        prev_element = element


def smoke_test_powder_standard():
    name = 'Si'
    cal = calibration_standards[name]
    assert(name == cal.name)

    for d, hkl, q in cal:
        assert_array_almost_equal(d_to_q(d), q)
        assert_array_almost_equal(q_to_d(q), d)
        assert_array_equal(np.linalg.norm(hkl), hkl.length)

    assert_equal(str(cal), "Calibration standard: Si")
    assert_equal(len(cal), 11)


def test_hkl():
    a = HKL(1, 1, 1)
    b = HKL('1', '1', '1')
    c = HKL(h='1', k='1', l='1')
    d = HKL(1.5, 1.5, 1.75)
    assert_equal(a, b)
    assert_equal(a, c)
    assert_equal(a, d)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
