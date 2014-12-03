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
from itertools import repeat
from skxray.core import q_to_d, d_to_q, twotheta_to_q, q_to_twotheta, verbosedict

_elm = namedtuple('_elm', ['Z', 'mass', 'rho', 'sym'])

_elm_lst = [_elm(1,  1.01,   9e-05,   'H'),
            _elm(2,  4.0,    0.00017, 'He'),
            _elm(3,  6.94,   0.534,   'Li'),
            _elm(4,  9.01,   1.85,    'Be'),
            _elm(5,  10.81,  2.34,    'B'),
            _elm(6,  12.01,  2.267,   'C'),
            _elm(7,  14.01,  0.00117, 'N'),
            _elm(8,  16.0,   0.00133, 'O'),
            _elm(9,  19.0,   0.0017,  'F'),
            _elm(10, 20.18,  0.00084, 'Ne'),
            _elm(11, 22.99,  0.97,    'Na'),
            _elm(12, 24.31,  1.741,   'Mg'),
            _elm(13, 26.98,  2.7,     'Al'),
            _elm(14, 28.09,  2.34,    'Si'),
            _elm(15, 30.97,  2.69,    'P'),
            _elm(16, 32.06,  2.08,    'S'),
            _elm(17, 35.45,  1.56,    'Cl'),
            _elm(18, 39.95,  0.00166, 'Ar'),
            _elm(19, 39.1,   0.86,    'K'),
            _elm(20, 40.08,  1.54,    'Ca'),
            _elm(21, 44.96,  3.0,     'Sc'),
            _elm(22, 47.9,   4.54,    'Ti'),
            _elm(23, 50.94,  6.1,     'V'),
            _elm(24, 52.0,   7.2,     'Cr'),
            _elm(25, 54.94,  7.44,    'Mn'),
            _elm(26, 55.85,  7.87,    'Fe'),
            _elm(27, 58.93,  8.9,     'Co'),
            _elm(28, 58.71,  8.908,   'Ni'),
            _elm(29, 63.55,  8.96,    'Cu'),
            _elm(30, 65.37,  7.14,    'Zn'),
            _elm(31, 69.72,  5.91,    'Ga'),
            _elm(32, 72.59,  5.323,   'Ge'),
            _elm(33, 74.92,  5.727,   'As'),
            _elm(34, 78.96,  4.81,    'Se'),
            _elm(35, 79.9,   3.1,     'Br'),
            _elm(36, 83.8,   0.00349, 'Kr'),
            _elm(37, 85.47,  1.53,    'Rb'),
            _elm(38, 87.62,  2.6,     'Sr'),
            _elm(39, 88.91,  4.6,     'Y'),
            _elm(40, 91.22,  6.5,     'Zr'),
            _elm(41, 92.91,  8.57,    'Nb'),
            _elm(42, 95.94,  10.2,    'Mo'),
            _elm(43, 98.91,  11.4,    'Tc'),
            _elm(44, 101.07, 12.4,    'Ru'),
            _elm(45, 102.91, 12.44,   'Rh'),
            _elm(46, 106.4,  12.0,    'Pd'),
            _elm(47, 107.87, 10.5,    'Ag'),
            _elm(48, 112.4,  8.65,    'Cd'),
            _elm(49, 114.82, 7.31,    'In'),
            _elm(50, 118.69, 7.3,     'Sn'),
            _elm(51, 121.75, 6.7,     'Sb'),
            _elm(52, 127.6,  6.24,    'Te'),
            _elm(53, 126.9,  4.94,    'I'),
            _elm(54, 131.3,  0.0055,  'Xe'),
            _elm(55, 132.9,  1.87,    'Cs'),
            _elm(56, 137.34, 3.6,     'Ba'),
            _elm(57, 138.91, 6.15,    'La'),
            _elm(58, 140.12, 6.8,     'Ce'),
            _elm(59, 140.91, 6.8,     'Pr'),
            _elm(60, 144.24, 6.96,    'Nd'),
            _elm(61, 145.0,  7.264,   'Pm'),
            _elm(62, 150.35, 7.5,     'Sm'),
            _elm(63, 151.96, 5.2,     'Eu'),
            _elm(64, 157.25, 7.9,     'Gd'),
            _elm(65, 158.92, 8.3,     'Tb'),
            _elm(66, 162.5,  8.5,     'Dy'),
            _elm(67, 164.93, 8.8,     'Ho'),
            _elm(68, 167.26, 9.0,     'Er'),
            _elm(69, 168.93, 9.3,     'Tm'),
            _elm(70, 173.04, 7.0,     'Yb'),
            _elm(71, 174.97, 9.8,     'Lu'),
            _elm(72, 178.49, 13.3,    'Hf'),
            _elm(73, 180.95, 16.6,    'Ta'),
            _elm(74, 183.85, 19.32,   'W'),
            _elm(75, 186.2,  20.5,    'Re'),
            _elm(76, 190.2,  22.48,   'Os'),
            _elm(77, 192.2,  22.42,   'Ir'),
            _elm(78, 195.09, 21.45,   'Pt'),
            _elm(79, 196.97, 19.3,    'Au'),
            _elm(80, 200.59, 13.59,   'Hg'),
            _elm(81, 204.37, 11.86,   'Tl'),
            _elm(82, 207.17, 11.34,   'Pb'),
            _elm(83, 208.98, 9.8,     'Bi'),
            _elm(84, 209.0,  9.2,     'Po'),
            _elm(85, 210.0,  6.4,     'At'),
            _elm(86, 222.0,  4.4,     'Rn'),
            _elm(87, 223.0,  2.9,     'Fr'),
            _elm(88, 226.0,  5.0,     'Ra'),
            _elm(89, 227.0,  10.1,    'Ac'),
            _elm(90, 232.04, 11.7,    'Th'),
            _elm(91, 231.0,  15.4,    'Pa'),
            _elm(92, 238.03, 19.1,    'U'),
            _elm(93, 237.0,  20.2,    'Np'),
            _elm(94, 244.0,  19.82,   'Pu'),
            _elm(95, 243.0,  12.0,    'Am'),
            _elm(96, 247.0,  13.51,   'Cm'),
            _elm(97, 247.0,  14.78,   'Bk'),
            _elm(98, 251.0,  15.1,    'Cf'),
            _elm(99, 252.0,  8.84,    'Es'),
            _elm(100, 257.0, np.nan,  'Fm')]

# make an empty dictionary
basic = dict()
# fill it with the data keyed on the symbol
basic.update((elm.sym.lower(), elm) for elm in _elm_lst)
# also add entries with it keyed on atomic number
basic.update((elm.Z, elm) for elm in _elm_lst)

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
        self._density = elem_dict.rho

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
