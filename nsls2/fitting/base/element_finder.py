'''
Copyright (c) 2014, Brookhaven National Laboratory
All rights reserved.

# @author: Li Li (lili@bnl.gov)
# created on 08/20/2014

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
import six
import numpy as np

from nsls2.fitting.base.element import Element



class ElementFinder(object):
    """
    Find emission lines close to a given energy

    Attributes
    ----------
    incident_e : float
        incident energy in KeV

    Methods
    -------
    find(self, energy, diff)
        return the possible lines close
        to a given energy value

    Examples
    --------
    >>> ef = ElementFinder(10)
    >>> out = ef.find(8, 0.5)
    >>> print (out)
    {'Eu': {'Lg4': 8.029999732971191}, 'Cu': {'Ka2': 8.027899742126465, 'Ka1': 8.047800064086914}}
    """

    def __init__(self, incident_e, **kwargs):
        """
        Parameters
        ----------
        incident_e : float
            incident energy in KeV
        kwargs : dict, option
            define element name,
            name1='Fe', name2='Cu'
            if not defined, search all elements
        """
        self._incident_e = incident_e

        if len(kwargs) == 0:
            self._search = 'all'
        else:
            self._search = kwargs.values()

    @property
    def incident_e(self):
        return self._incident_e

    @incident_e.setter
    def incident_e(self, val):
        """
        Parameters
        ----------
        val : float
            new incident energy value in KeV
        """
        self._incident_e = float(val)


    def find(self, energy, diff):
        """
        Parameters
        ----------
        energy : float
            energy value to search for
        diff : float
            difference compared to energy

        Returns
        -------
        result : dict
            elements and possible lines
        """

        result = {}
        if self._search == 'all':
            for i in np.arange(100):
                e = Element(i+1, self._incident_e)
                if find_line(e, energy, diff) is None:
                    continue
                result.update(find_line(e, energy, diff))
        else:
            for item in self._search:
                e = Element(item, self._incident_e)
                if find_line(e, energy, diff) is None:
                    continue
                result.update(find_line(e, energy, diff))

        return result



def find_line(element, energy, diff):
    """
    Fine possible line from a given element

    Parameters
    ----------
    element : class instance
        instance of Element
    energy : float
        energy value to search for
    diff : float
        define search range (energy - diff, energy + diff)

    Returns
    -------
    dict or None
        elements with associated lines
    """
    mydict = {k : v for k, v in six.iteritems(element.emission_line) if abs(v - energy) < diff}
    if len(mydict) == 0:
        return
    else:
        newdict = {k : v for k, v in six.iteritems(mydict) if element.cs[k] > 0}
        if len(newdict) == 0:
            return
        else:
            return {element.name: newdict}
