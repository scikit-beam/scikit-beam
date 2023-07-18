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
from __future__ import absolute_import, division, print_function

import functools
import logging
import os
from collections import namedtuple

import six

logger = logging.getLogger(__name__)


data_dir = os.path.join(os.path.dirname(__file__), "data")


element = namedtuple(
    "element",
    [
        "Z",
        "sym",
        "name",
        "atomic_radius",
        "covalent_radius",
        "mass",
        "bp",
        "mp",
        "density",
        "atomic_volume",
        "coherent_scattering_length",
        "incoherent_crosssection",
        "absorption",
        "debye_temp",
        "thermal_conductivity",
    ],
)


def read_atomic_constants():
    """Returns a dictionary of atomic constants

    Returns
    -------
    constants : dict
        keys: ['Z'. 'sym', 'name', 'atomic_radius', 'covalent_radius', 'mass',
               'bp', 'mp', 'density', 'atomic_volume',
               'coherent_scattering_length', 'incoherent_crosssection',
               'absorption', 'debye_temp', 'thermal_conductivity']
    """
    basic = {}
    field_desc = []
    with open(os.path.join(data_dir, "AtomicConstants.dat"), "r") as infile:
        for line in infile:
            if line.split()[0] == "#S":
                s = line.split()
                abbrev = s[2]
                Z = int(s[1])
                if Z == 1000:
                    break
            elif not field_desc and line.split()[0] == "#L":
                field_desc = [
                    "Atomic number",
                    "Element symbol (Fe, Cr, etc.)",
                    "Full element name (Iron, Chromium, etc.",
                ]
                field_desc += line.split()[1:]
            elif line.startswith("#UNAME"):
                elem_name = line.split()[1]
            elif line[0] == "#":
                continue
            else:
                data = [float(item) for item in line.split()]
                data = [Z, abbrev, elem_name] + data
                elem = element(*data)
                basic[abbrev.lower()] = elem
    return basic, field_desc


basic, field_descriptors = read_atomic_constants()
# also add entries with it keyed on atomic number
basic.update({elm.Z: elm for elm in six.itervalues(basic)})
basic.update({elm.name.lower(): elm for elm in six.itervalues(basic)})
basic.update({elm.sym.lower(): elm for elm in six.itervalues(basic)})
doc_title = """
    Object to return basic elemental information
    """
doc_params = """
    element : str or int
        Element symbol, name or atomic number ('Zinc', 'Zn' or 30)
    """
fields = ["Z : int", "sym : str", "name : str"] + ["{} : float".format(field) for field in element._fields[3:]]

fields = ["{}\n        {}".format(field, field_desc) for field, field_desc in zip(fields, field_descriptors)]

doc_attrs = "\n    " + "\n    ".join(fields)

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
    """.format(
        doc_title, doc_params, doc_attrs, doc_ex
    )

    def __init__(self, Z, *args, **kwargs):
        # init the parent object
        super(BasicElement, self).__init__(*args, **kwargs)
        # bash the element abbreviation down to lowercase
        if isinstance(Z, six.string_types):
            Z = Z.lower()
        # stash the element tuple
        self._element = basic[Z]
        # set the class attributes
        for e in element._fields:
            setattr(self, e, getattr(basic[Z], e))

    # allow the Element to work as a dictionary as well
    def __getitem__(self, item):
        return getattr(self, item)

    def __repr__(self):
        return six.text_type("BasicElement({})".format(self.Z))

    # pretty print the element
    def __str__(self):
        desc = self.name + "\n" + "=" * len(self.name)
        for d in dir(self):
            if d.startswith("_"):
                continue
            desc += "\n{}: {}".format(d, getattr(self, d))

        return desc

    def __eq__(self, other):
        return self.Z == other.Z

    def __lt__(self, other):
        return self.Z < other.Z
