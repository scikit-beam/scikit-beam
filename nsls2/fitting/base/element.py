'''
Copyright (c) 2014, Brookhaven National Laboratory
All rights reserved.

# @author: Li Li (lili@bnl.gov)
# created on 08/16/2014

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the Brookhaven National Laboratory nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

from __future__ import (absolute_import, division)
from collections import Mapping
import six

from nsls2.fitting.base.element_data import (XRAYLIB_MAP, OTHER_VAL)


class Element(object):
    """
    Object to return all the elemental information
    related to fluorescence

    Attributes
    ----------
    name : str
        element name, such as Fe, Cu
    z : int
        atomic number
    mass : float
        atomic mass in g/mol
    density : float
        element density in g/cm3
    energy : float
        incident energy in KeV
    emission_line[line] : float
        energy of emission line
        line is string type and defined as 'Ka1', 'Kb1'.
        unit in KeV
    cs[line] : float
        scattering cross section
        line is string type and defined as 'Ka1', 'Kb1'.
        unit in cm2/g
    bind_energy[shell] : float
        binding energy
        shell is string type and defined as "K", "L1".
        unit in KeV
    jump_factor[shell] : float
        jump factor
        shell is string type and defined as "K", "L1".
    f_yield[shell] : float
        fluorescence yield
        shell is string type and defined as "K", "L1".

    Examples
    --------
    >>> e = Element('Zn', 10) # or e = Element(30, 10)
    >>> print (e.emission_line['Ka1']) # energy for emission line Ka1
    >>> print (e.cs['Ka1']) # cross section for emission line Ka1
    >>> print (e.cs.items()) # output all the cross section
    >>> print (e.f_yield['K']) # fluorescence yield for K shell
    >>> print (e.mass) #atomic mass
    >>> print (e.density) #density
    """
    def __init__(self, element, energy):
        """
        Parameters
        ----------
        element : int or str
            element name or element atomic Z
        energy : float
            incident x-ray energy, in KeV
        """
        try:
            # forcibly down-cast stringy inputs to lowercase
            if isinstance(element, six.string_types):
                element = element.lower()
            elem_dict = OTHER_VAL[element]
        except KeyError:
            raise ValueError('Please define element by '
                             'atomic number z or element name')

        self.name = elem_dict['sym']
        self.z = elem_dict['Z']
        self.mass = elem_dict['mass']
        self.density = elem_dict['rho']
        self._element = self.z

        self._energy = float(energy)

        self.emission_line = _XrayLibWrap('lines', self._element)
        self.cs = _XrayLibWrap('cs', self._element, energy)
        self.bind_energy = _XrayLibWrap('binding_e', self._element)
        self.jump_factor = _XrayLibWrap('jump', self._element)
        self.f_yield = _XrayLibWrap('yield', self._element)

    @property
    def element(self):
        return self._element

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, val):
        """
        Parameters
        ----------
        val : float
            new energy value in KeV
        """
        if not isinstance(val, float and int):
            raise TypeError('Expected a number for energy')
        self._energy = val
        self.cs.energy = val

    def __repr__(self):
        return 'Element name %s with atomic Z %s' % (self.name, self.z)


class _XrayLibWrap(Mapping):
    """
    This is an interface to wrap xraylib to perform calculation related
    to xray fluorescence.

    Attributes
    ----------
    info_type : str
        option to choose which physics quantity to calculate
    element : int
        atomic number
    energy : float, optional
        incident energy for fluorescence in KeV
    """
    def __init__(self, info_type,
                 element, energy=None):
        self.info_type = info_type
        self._map, self._func = XRAYLIB_MAP[info_type]
        self._keys = sorted(list(six.iterkeys(self._map)))
        self._element = element
        self._energy = energy

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, val):
        """
        Parameters
        ----------
        val : float
            new energy value
        """
        self._energy = val

    def __getitem__(self, key):
        """
        call xraylib function to calculate physics quantity

        Parameters
        ----------
        key : str
            defines which physics quantity to calculate
        """
        if self.info_type == 'cs':
            return self._func(self._element,
                              self._map[key.lower()],
                              self.energy)
        else:
            return self._func(self._element,
                              self._map[key.lower()])

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)
