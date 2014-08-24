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
import numpy as np
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
    emission_line[line] : float
        energy of emission line
        line is string type and defined as 'Ka1', 'Kb1'.
        unit in KeV
    cs(energy)[line] : float
        scattering cross section
        energy is incident energy
        line is string type and defined as 'Ka1', 'Kb1'.
        unit in cm2/g
    bind_energy[shell] : float
        binding energy
        shell is string type and defined as "K", "L1".
        unit in KeV
    jump_factor[shell] : float
        jump factor
        shell is string type and defined as "K", "L1".
    fluor_yield[shell] : float
        fluorescence yield
        shell is string type and defined as "K", "L1".

    Methods
    -------
    find(energy, delta_e)
        find possible emission lines close to a energy

    Examples
    --------
    >>> e = Element('Zn') # or e = Element(30), 30 is atomic number
    >>> print (e.emission_line['Ka1']) # energy for emission line Ka1
    >>> print (e.cs(10)['Ka1']) # cross section for emission line Ka1, 10 is incident energy
    >>> print (e.cs.items()) # output all the cross section
    >>> print (e.fluor_yield['K']) # fluorescence yield for K shell
    >>> print (e.mass) #atomic mass
    >>> print (e.density) #density
    >>> print (e.find(10, 0.5)) #emission lines within range(10 - 0.5, 10 + 0.5)
    """
    def __init__(self, element):
        """
        Parameters
        ----------
        element : int or str
            element name or element atomic Z
        """

        # forcibly down-cast stringy inputs to lowercase
        if isinstance(element, six.string_types):
            element = element.lower()
        elem_dict = OTHER_VAL[element]

        self._name = elem_dict['sym']
        self._z = elem_dict['Z']
        self._mass = elem_dict['mass']
        self._density = elem_dict['rho']

        self._emission_line = _XrayLibWrap('lines', self._z)
        self._bind_energy = _XrayLibWrap('binding_e', self._z)
        self._jump_factor = _XrayLibWrap('jump', self._z)
        self._fluor_yield = _XrayLibWrap('yield', self._z)

    @property
    def name(self):
        return self._name

    @property
    def z(self):
        return self._z

    @property
    def mass(self):
        return self._mass

    @property
    def density(self):
        return self._density

    @property
    def emission_line(self):
        return self._emission_line

    @property
    def cs(self):
        """
        Returns
        -------
        function:
            function with incident energy as argument
        """
        def myfunc(incident_energy):
            return _XrayLibWrap('cs', self.z, incident_energy)
        return myfunc

    @property
    def bind_energy(self):
        return self._bind_energy

    @property
    def jump_factor(self):
        return self._jump_factor

    @property
    def fluor_yield(self):
        return self._fluor_yield

    def __repr__(self):
        return 'Element name %s with atomic Z %s' % (self.name, self.z)

    def line_near(self, energy, delta_e):
        """
        Fine possible lines from the element

        Parameters
        ----------
        energy : float
            energy value to search for
        delta_e : float
            define search range (energy - delta_e, energy + delta_e)

        Returns
        -------
        dict
            all possible emission lines
        """
        out_dict = dict()
        for k, v in six.iteritems(self.emission_line):
            if self.cs[k] == 0:
                continue
            if np.abs(v - energy) < delta_e:
                out_dict[k] = v
        return out_dict



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
    incident_energy : float, optional
        incident energy for fluorescence in KeV
    """
    def __init__(self, info_type,
                 element, incident_energy=None):
        self.info_type = info_type
        self._map, self._func = XRAYLIB_MAP[info_type]
        self._keys = sorted(list(six.iterkeys(self._map)))
        self._element = element
        self._incident_energy = incident_energy

    @property
    def incident_energy(self):
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
        call xraylib function to calculate physics quantity

        Parameters
        ----------
        key : str
            defines which physics quantity to calculate
        """
        if self.info_type == 'cs':
            return self._func(self._element,
                              self._map[key.lower()],
                              self._incident_energy)
        else:
            return self._func(self._element,
                              self._map[key.lower()])

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)
