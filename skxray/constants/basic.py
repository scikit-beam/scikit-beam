# -*- coding: utf-8 -*-
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

from __future__ import (absolute_import, division,
                        unicode_literals, print_function)
import numpy as np
import six
from collections import Mapping, namedtuple
import functools
import os
from itertools import repeat
from skxray.core import (q_to_d, d_to_q, twotheta_to_q, q_to_twotheta,
                         verbosedict)


data_dir = os.path.join(os.path.dirname(__file__), 'data')


element = namedtuple('element',
                     ['Z', 'sym', 'name', 'atomic_radius',
                      'covalent_radius', 'mass', 'bp', 'mp', 'density',
                      'atomic_volume', 'coherent_scattering_length',
                      'incoherent_crosssection', 'absorption', 'debye_temp',
                      'thermal_conductivity']
)


def read_atomic_constants():
    """Returns Atomic Constants

    Array is in format:
    0 = Atomic Number
    1 = Atomic Radius [A]
    2 = CovalentRadius [A]
    3 = AtomicMass
    4 = BoilingPoint [K]
    5 = MeltingPoint [K]
    6 = Density [g/ccm]
    7 = Atomic Volume
    8 = CoherentScatteringLength [1E-12cm]
    9 = IncoherentX-section [barn]
    10 = Absorption@1.8A [barn]
    11 = DebyeTemperature [K]
    12 = ThermalConductivity [W/cmK]

    """
    basic = {}
    with open(os.path.join(data_dir, 'AtomicConstants.dat'),'r') as infile:
        for line in infile:
            if line.split()[0] == '#S':
                s = line.split()
                abbrev = s[2]
                Z = int(s[1])
                if Z == 1000:
                    break
            elif line.startswith('#UNAME'):
                elem_name = line.split()[1]
            elif line[0] == '#':
                continue
            else:
                data = [float(item) for item in line.split()]
                data = [Z, abbrev, elem_name] + data
                elem = element(*data)
                basic[abbrev.lower()] = elem
    return basic

basic = read_atomic_constants()
# also add entries with it keyed on atomic number
basic.update({elm.Z: elm for elm in six.itervalues(basic)})

doc_title = """
    Object to return basic elemental information
    """
doc_params = """
    element : str or int
        Element symbol or element atomic Z
    """
doc_attrs = """
    name : str
    Z ; int
    mass : float
    density : float
    """
doc_ex = """
    >>> # Create an `Element` object
    >>> e = Element('Zn') # or e = Element(30)
    >>> # get the atomic mass
    >>> e.mass
    65.37
    >>> # get the density in grams / cm^3
    >>> e.density
    7.14
    """

@functools.total_ordering
class BasicElement(object):
    # define the docs
    __doc__ = """{}
    Parameters
    ----------{}
    Attributes
    ----------{}
    Examples
    --------{}
    """.format(doc_title,
               doc_params,
               doc_attrs,
               doc_ex)

    def __init__(self, element):
        if isinstance(element, six.string_types):
            element = element.lower()
        elem_dict = basic[element]

        self._name = elem_dict.sym
        self._z = elem_dict.Z
        self._mass = elem_dict.mass
        self._density = elem_dict.density

    @property
    def name(self):
        """
        Atomic symbol, `str`

        such as Fe, Cu
        """
        return self._name

    @property
    def Z(self):
        """
        atomic number, `int`
        """
        return self._z

    @property
    def mass(self):
        """
        atomic mass in g/mol, `float`
        """
        return self._mass

    @property
    def density(self):
        """
        element density in g/cm3, `float`
        """
        return self._density

    def __repr__(self):
        return 'Element name %s with atomic Z %s' % (self.name, self._z)

    def __eq__(self, other):
        return self.Z == other.Z

    def __lt__(self, other):
        return self.Z < other.Z
