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
from numpy.testing import assert_equal

from skbeam.core.constants.basic import BasicElement, basic, element


def test_element_creation():
    # grab the set of elements represented by 'Z'
    elements = sorted([elm for abbrev, elm in six.iteritems(basic) if isinstance(abbrev, int)])

    for e in elements:
        sym = e.sym
        name = e.name
        # make sure that the elements can be initialized with Z or any
        # combination of element symbols or the element name
        inits = [sym, sym.upper(), sym.lower(), sym.swapcase(), name, name.upper(), name.lower(), name.swapcase()]
        # loop over the initialization routines to smoketest element creation
        for init in inits:
            elem = BasicElement(init)

        # create an element with the Z value
        elem = BasicElement(e.Z)
        str(elem)
        # obtain all attribute fields of the Element to ensure it is
        # behaving correctly
        for field in element._fields:
            tuple_attr = getattr(basic[e.Z], field)
            elem_attr_dct = elem[str(field)]
            elem_attr = getattr(elem, field)
            # shield the assertion from any elements whose density is
            # unknown
            try:
                if np.isnan(tuple_attr):
                    continue
            except TypeError:
                pass
            assert_equal(elem_attr_dct, tuple_attr)
            assert_equal(elem_attr, tuple_attr)
            assert_equal(elem_attr_dct, elem_attr)

    # test the comparators
    for e1, e2 in zip(elements, elements[1:]):
        # compare prev_element to element
        assert_equal(e1.__lt__(e2), True)
        assert_equal(e1 < e2, True)
        assert_equal(e1.__eq__(e2), False)
        assert_equal(e1 == e2, False)
        assert_equal(e1 >= e2, False)
        assert_equal(e1 > e2, False)
        # compare element to prev_element
        assert_equal(e2 < e1, False)
        assert_equal(e2.__lt__(e1), False)
        assert_equal(e2 <= e1, False)
        assert_equal(e2.__eq__(e1), False)
        assert_equal(e2 == e1, False)
        assert_equal(e2 >= e1, True)
        assert_equal(e2 > e1, True)
