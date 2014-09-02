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


from __future__ import (absolute_import, division)
from collections import Mapping
import numpy as np
import six
import functools

from nsls2.fitting.base.element_data import (XRAYLIB_MAP, OTHER_VAL)


@functools.total_ordering
class Element(object):
    """
    Object to return all the elemental information
    related to fluorescence

    Attributes
    ----------
    name
    Z
    mass
    density
    emission_line : float
        Emission line can be used as a unique characteristic
        for qualitative identification of the element.
        line is string type and defined as 'Ka1', 'Kb1'.
        unit in KeV
    cs : float
        Fluorescence cross section
        energy is incident energy
        line is string type and defined as 'Ka1', 'Kb1'.
        unit in cm2/g
    bind_energy : float
        Binding energy is a measure of the energy required
        to free electrons from their atomic orbits.
        shell is string type and defined as "K", "L1".
        unit in KeV
    jump_factor : float
        Absorption jump factor is defined as the fraction
        of the total absorption that is associated with
        a given shell rather than for any other shell.
        shell is string type and defined as "K", "L1".
    fluor_yield : float
        The fluorescence quantum yield gives the efficiency
        of the fluorescence process, and is defined as the ratio of the
        number of photons emitted to the number of photons absorbed.
        shell is string type and defined as "K", "L1".

    Methods
    -------
    line_near(energy, delta_e, incident_energy)
        Find possible emission lines close to a energy at given
        incident_energy

    Parameters
    ----------
    element : int or str
        Element name or element atomic Z

    Examples
    --------
    >>> e = Element('Zn') # or e = Element(30), 30 is atomic number
    >>> e.emission_line['Ka1'] # energy for emission line Ka1
    8.638900756835938
    >>> e.cs(10)['Ka1'] # cross section for emission line Ka1, 10 is incident energy
    54.756561279296875
    >>> e.fluor_yield['K'] # fluorescence yield for K shell
    0.46936899423599243
    >>> e.mass #atomic mass
    65.37
    >>> e.density #density
    7.14
    >>> e.find(10, 0.5, 12) #emission lines within range(10 - 0.5, 10 + 0.5) at incident 12 KeV
    {'kb1': 9.571999549865723}
    #########################   useful command   ###########################
    >>> e.emission_line.all # list all the emission lines
    [('ka1', 8.638900756835938),
     ('ka2', 8.615799903869629),
     ('kb1', 9.571999549865723),
     ('kb2', 0.0),
     ('la1', 1.0116000175476074),
     ('la2', 1.0116000175476074),
     ('lb1', 1.0346999168395996),
     ('lb2', 0.0),
     ('lb3', 1.1069999933242798),
     ('lb4', 1.1069999933242798),
     ('lb5', 0.0),
     ('lg1', 0.0),
     ('lg2', 0.0),
     ('lg3', 0.0),
     ('lg4', 0.0),
     ('ll', 0.8837999701499939),
     ('ln', 0.9069000482559204),
     ('ma1', 0.0),
     ('ma2', 0.0),
     ('mb', 0.0),
     ('mg', 0.0)]
    >>> e.cs(10).all # list all the emission lines
    [('ka1', 54.756561279296875),
     ('ka2', 28.13692855834961),
     ('kb1', 7.509212970733643),
     ('kb2', 0.0),
     ('la1', 0.13898827135562897),
     ('la2', 0.01567710004746914),
     ('lb1', 0.0791187509894371),
     ('lb2', 0.0),
     ('lb3', 0.004138986114412546),
     ('lb4', 0.002259803470224142),
     ('lb5', 0.0),
     ('lg1', 0.0),
     ('lg2', 0.0),
     ('lg3', 0.0),
     ('lg4', 0.0),
     ('ll', 0.008727769367396832),
     ('ln', 0.00407258840277791),
     ('ma1', 0.0),
     ('ma2', 0.0),
     ('mb', 0.0),
     ('mg', 0.0)]
    """
    def __init__(self, element):

        # forcibly down-cast stringy inputs to lowercase
        if isinstance(element, six.string_types):
            element = element.lower()
        elem_dict = OTHER_VAL[element]

        self._name = elem_dict['sym']
        self._z = elem_dict['Z']
        self._mass = elem_dict['mass']
        self._density = elem_dict['rho']

        self._emission_line = XrayLibWrap(self._z, 'lines')
        self._bind_energy = XrayLibWrap(self._z, 'binding_e')
        self._jump_factor = XrayLibWrap(self._z, 'jump')
        self._fluor_yield = XrayLibWrap(self._z, 'yield')

    @property
    def name(self):
        """element name, such as Fe, Cu"""
        return self._name

    @property
    def Z(self):
        """atomic number"""
        return self._z

    @property
    def mass(self):
        """atomic mass in g/mol"""
        return self._mass

    @property
    def density(self):
        """element density in g/cm3"""
        return self._density

    @property
    def emission_line(self):
        """
        unique characteristic for qualitative identification of the element, defined as 'Ka1', 'Kb1'.
        """
        return self._emission_line

    @property
    def cs(self):
        """fluorescence cross section in cm2/g, defined as 'Ka1', 'Kb1'."""
        def myfunc(incident_energy):
            return XrayLibWrap_Energy(self._z, 'cs',
                                      incident_energy)
        return myfunc

    @property
    def bind_energy(self):
        """measure of the energy required to free electrons from their atomic orbits, in KeV"""
        return self._bind_energy

    @property
    def jump_factor(self):
        """
        the fraction of the total absorption that is associated with a given shell rather than for any other shell.
        """
        return self._jump_factor

    @property
    def fluor_yield(self):
        """the ratio of the number of photons emitted to the number of photons absorbed."""
        return self._fluor_yield

    def __repr__(self):
        return 'Element name %s with atomic Z %s' % (self.name, self._z)

    def __eq__(self, other):
        return self.Z == other.Z

    def __lt__(self, other):
        return self.Z < other.Z

    def line_near(self, energy, delta_e,
                  incident_energy):
        """
        Find possible emission lines given the element.

        Parameters
        ----------
        energy : float
            Energy value to search for
        delta_e : float
            Define search range (energy - delta_e, energy + delta_e)
        incident_energy : float
            incident energy of x-ray in KeV

        Returns
        -------
        dict
            all possible emission lines
        """
        out_dict = dict()
        for k, v in six.iteritems(self.emission_line):
            if self.cs(incident_energy)[k] == 0:
                continue
            if np.abs(v - energy) < delta_e:
                out_dict[k] = v
        return out_dict


class XrayLibWrap(Mapping):
    """
    This is an interface to wrap xraylib to perform calculation related
    to xray fluorescence.

    Attributes
    ----------
    all : list
        List the physics quantity for
        all the lines or all the shells.

    Parameters
    ----------
    element : int
        atomic number
    info_type : str
        option to choose which physics quantity to calculate as follows
        lines : emission lines
        bind_e : binding energy
        jump : absorption jump factor
        yield : fluorescence yield
    """
    def __init__(self, element, info_type):
        self._element = element
        self.info_type = info_type
        self._map, self._func = XRAYLIB_MAP[info_type]
        self._keys = sorted(list(six.iterkeys(self._map)))

    @property
    def all(self):
        """List the physics quantity for all the lines or all the shells. """
        return list(six.iteritems(self))

    def __getitem__(self, key):
        """
        Call xraylib function to calculate physics quantity.

        Parameters
        ----------
        key : str
            Define which physics quantity to calculate.
        """

        return self._func(self._element,
                          self._map[key.lower()])

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)


class XrayLibWrap_Energy(XrayLibWrap):
    """
    This is an interface to wrap xraylib
    to perform calculation on fluorescence
    cross section, or other incident energy
    related quantity.

    Attributes
    ----------
    incident_energy : float
        incident energy for fluorescence in KeV

    Parameters
    ----------
    element : int
        atomic number
    info_type : str
        option to calculate physics quantity
        related to incident energy, such as
        cs : cross section
    incident_energy : float
        incident energy for fluorescence in KeV
    """
    def __init__(self, element, info_type, incident_energy):
        super(XrayLibWrap_Energy, self).__init__(element, info_type)
        self._incident_energy = incident_energy

    @property
    def incident_energy(self):
        """incident energy for fluorescence in KeV"""
        return self._incident_energy

    @incident_energy.setter
    def incident_energy(self, val):
        """
        Parameters
        ----------
        val : float
            new energy value
        """
        self._incident_energy = val

    def __getitem__(self, key):
        """
        Call xraylib function to calculate physics quantity.

        Parameters
        ----------
        key : str
            defines which physics quantity to calculate
        """
        return self._func(self._element,
                          self._map[key.lower()],
                          self._incident_energy)
