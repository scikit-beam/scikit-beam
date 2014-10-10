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
from nsls2.core import q_to_d, d_to_q, twotheta_to_q, q_to_twotheta, verbosedict

import xraylib
xraylib.XRayInit()
xraylib.SetErrorMessages(0)


line_name = ['Ka1', 'Ka2', 'Kb1', 'Kb2', 'La1', 'La2', 'Lb1', 'Lb2',
             'Lb3', 'Lb4', 'Lb5', 'Lg1', 'Lg2', 'Lg3', 'Lg4', 'Ll',
             'Ln', 'Ma1', 'Ma2', 'Mb', 'Mg']
line_list = [xraylib.KA1_LINE, xraylib.KA2_LINE, xraylib.KB1_LINE,
             xraylib.KB2_LINE, xraylib.LA1_LINE, xraylib.LA2_LINE,
             xraylib.LB1_LINE, xraylib.LB2_LINE, xraylib.LB3_LINE,
             xraylib.LB4_LINE, xraylib.LB5_LINE, xraylib.LG1_LINE,
             xraylib.LG2_LINE, xraylib.LG3_LINE, xraylib.LG4_LINE,
             xraylib.LL_LINE, xraylib.LE_LINE, xraylib.MA1_LINE,
             xraylib.MA2_LINE, xraylib.MB_LINE, xraylib.MG_LINE]

line_dict = verbosedict((k.lower(), v) for k, v in zip(line_name, line_list))


bindingE = ['K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5', 'N1',
            'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'O1', 'O2', 'O3',
            'O4', 'O5', 'P1', 'P2', 'P3']

shell_list = [xraylib.K_SHELL, xraylib.L1_SHELL, xraylib.L2_SHELL,
              xraylib.L3_SHELL, xraylib.M1_SHELL, xraylib.M2_SHELL,
              xraylib.M3_SHELL, xraylib.M4_SHELL, xraylib.M5_SHELL,
              xraylib.N1_SHELL, xraylib.N2_SHELL, xraylib.N3_SHELL,
              xraylib.N4_SHELL, xraylib.N5_SHELL, xraylib.N6_SHELL,
              xraylib.N7_SHELL, xraylib.O1_SHELL, xraylib.O2_SHELL,
              xraylib.O3_SHELL, xraylib.O4_SHELL, xraylib.O5_SHELL,
              xraylib.P1_SHELL, xraylib.P2_SHELL, xraylib.P3_SHELL]

shell_dict = verbosedict((k.lower(), v) for k, v in zip(bindingE, shell_list))


XRAYLIB_MAP = verbosedict({'lines': (line_dict, xraylib.LineEnergy),
                           'cs': (line_dict, xraylib.CS_FluorLine),
                           'binding_e': (shell_dict, xraylib.EdgeEnergy),
                           'jump': (shell_dict, xraylib.JumpFactor),
                           'yield': (shell_dict, xraylib.FluorYield),
                           })

elm_data_list = [{'Z': 1, 'mass': 1.01, 'rho': 9e-05, 'sym': 'H'},
                 {'Z': 2, 'mass': 4.0, 'rho': 0.00017, 'sym': 'He'},
                 {'Z': 3, 'mass': 6.94, 'rho': 0.534, 'sym': 'Li'},
                 {'Z': 4, 'mass': 9.01, 'rho': 1.85, 'sym': 'Be'},
                 {'Z': 5, 'mass': 10.81, 'rho': 2.34, 'sym': 'B'},
                 {'Z': 6, 'mass': 12.01, 'rho': 2.267, 'sym': 'C'},
                 {'Z': 7, 'mass': 14.01, 'rho': 0.00117, 'sym': 'N'},
                 {'Z': 8, 'mass': 16.0, 'rho': 0.00133, 'sym': 'O'},
                 {'Z': 9, 'mass': 19.0, 'rho': 0.0017, 'sym': 'F'},
                 {'Z': 10, 'mass': 20.18, 'rho': 0.00084, 'sym': 'Ne'},
                 {'Z': 11, 'mass': 22.99, 'rho': 0.97, 'sym': 'Na'},
                 {'Z': 12, 'mass': 24.31, 'rho': 1.741, 'sym': 'Mg'},
                 {'Z': 13, 'mass': 26.98, 'rho': 2.7, 'sym': 'Al'},
                 {'Z': 14, 'mass': 28.09, 'rho': 2.34, 'sym': 'Si'},
                 {'Z': 15, 'mass': 30.97, 'rho': 2.69, 'sym': 'P'},
                 {'Z': 16, 'mass': 32.06, 'rho': 2.08, 'sym': 'S'},
                 {'Z': 17, 'mass': 35.45, 'rho': 1.56, 'sym': 'Cl'},
                 {'Z': 18, 'mass': 39.95, 'rho': 0.00166, 'sym': 'Ar'},
                 {'Z': 19, 'mass': 39.1, 'rho': 0.86, 'sym': 'K'},
                 {'Z': 20, 'mass': 40.08, 'rho': 1.54, 'sym': 'Ca'},
                 {'Z': 21, 'mass': 44.96, 'rho': 3.0, 'sym': 'Sc'},
                 {'Z': 22, 'mass': 47.9, 'rho': 4.54, 'sym': 'Ti'},
                 {'Z': 23, 'mass': 50.94, 'rho': 6.1, 'sym': 'V'},
                 {'Z': 24, 'mass': 52.0, 'rho': 7.2, 'sym': 'Cr'},
                 {'Z': 25, 'mass': 54.94, 'rho': 7.44, 'sym': 'Mn'},
                 {'Z': 26, 'mass': 55.85, 'rho': 7.87, 'sym': 'Fe'},
                 {'Z': 27, 'mass': 58.93, 'rho': 8.9, 'sym': 'Co'},
                 {'Z': 28, 'mass': 58.71, 'rho': 8.908, 'sym': 'Ni'},
                 {'Z': 29, 'mass': 63.55, 'rho': 8.96, 'sym': 'Cu'},
                 {'Z': 30, 'mass': 65.37, 'rho': 7.14, 'sym': 'Zn'},
                 {'Z': 31, 'mass': 69.72, 'rho': 5.91, 'sym': 'Ga'},
                 {'Z': 32, 'mass': 72.59, 'rho': 5.323, 'sym': 'Ge'},
                 {'Z': 33, 'mass': 74.92, 'rho': 5.727, 'sym': 'As'},
                 {'Z': 34, 'mass': 78.96, 'rho': 4.81, 'sym': 'Se'},
                 {'Z': 35, 'mass': 79.9, 'rho': 3.1, 'sym': 'Br'},
                 {'Z': 36, 'mass': 83.8, 'rho': 0.00349, 'sym': 'Kr'},
                 {'Z': 37, 'mass': 85.47, 'rho': 1.53, 'sym': 'Rb'},
                 {'Z': 38, 'mass': 87.62, 'rho': 2.6, 'sym': 'Sr'},
                 {'Z': 39, 'mass': 88.91, 'rho': 4.6, 'sym': 'Y'},
                 {'Z': 40, 'mass': 91.22, 'rho': 6.5, 'sym': 'Zr'},
                 {'Z': 41, 'mass': 92.91, 'rho': 8.57, 'sym': 'Nb'},
                 {'Z': 42, 'mass': 95.94, 'rho': 10.2, 'sym': 'Mo'},
                 {'Z': 43, 'mass': 98.91, 'rho': 11.4, 'sym': 'Tc'},
                 {'Z': 44, 'mass': 101.07, 'rho': 12.4, 'sym': 'Ru'},
                 {'Z': 45, 'mass': 102.91, 'rho': 12.44, 'sym': 'Rh'},
                 {'Z': 46, 'mass': 106.4, 'rho': 12.0, 'sym': 'Pd'},
                 {'Z': 47, 'mass': 107.87, 'rho': 10.5, 'sym': 'Ag'},
                 {'Z': 48, 'mass': 112.4, 'rho': 8.65, 'sym': 'Cd'},
                 {'Z': 49, 'mass': 114.82, 'rho': 7.31, 'sym': 'In'},
                 {'Z': 50, 'mass': 118.69, 'rho': 7.3, 'sym': 'Sn'},
                 {'Z': 51, 'mass': 121.75, 'rho': 6.7, 'sym': 'Sb'},
                 {'Z': 52, 'mass': 127.6, 'rho': 6.24, 'sym': 'Te'},
                 {'Z': 53, 'mass': 126.9, 'rho': 4.94, 'sym': 'I'},
                 {'Z': 54, 'mass': 131.3, 'rho': 0.0055, 'sym': 'Xe'},
                 {'Z': 55, 'mass': 132.9, 'rho': 1.87, 'sym': 'Cs'},
                 {'Z': 56, 'mass': 137.34, 'rho': 3.6, 'sym': 'Ba'},
                 {'Z': 57, 'mass': 138.91, 'rho': 6.15, 'sym': 'La'},
                 {'Z': 58, 'mass': 140.12, 'rho': 6.8, 'sym': 'Ce'},
                 {'Z': 59, 'mass': 140.91, 'rho': 6.8, 'sym': 'Pr'},
                 {'Z': 60, 'mass': 144.24, 'rho': 6.96, 'sym': 'Nd'},
                 {'Z': 61, 'mass': 145.0, 'rho': 7.264, 'sym': 'Pm'},
                 {'Z': 62, 'mass': 150.35, 'rho': 7.5, 'sym': 'Sm'},
                 {'Z': 63, 'mass': 151.96, 'rho': 5.2, 'sym': 'Eu'},
                 {'Z': 64, 'mass': 157.25, 'rho': 7.9, 'sym': 'Gd'},
                 {'Z': 65, 'mass': 158.92, 'rho': 8.3, 'sym': 'Tb'},
                 {'Z': 66, 'mass': 162.5, 'rho': 8.5, 'sym': 'Dy'},
                 {'Z': 67, 'mass': 164.93, 'rho': 8.8, 'sym': 'Ho'},
                 {'Z': 68, 'mass': 167.26, 'rho': 9.0, 'sym': 'Er'},
                 {'Z': 69, 'mass': 168.93, 'rho': 9.3, 'sym': 'Tm'},
                 {'Z': 70, 'mass': 173.04, 'rho': 7.0, 'sym': 'Yb'},
                 {'Z': 71, 'mass': 174.97, 'rho': 9.8, 'sym': 'Lu'},
                 {'Z': 72, 'mass': 178.49, 'rho': 13.3, 'sym': 'Hf'},
                 {'Z': 73, 'mass': 180.95, 'rho': 16.6, 'sym': 'Ta'},
                 {'Z': 74, 'mass': 183.85, 'rho': 19.32, 'sym': 'W'},
                 {'Z': 75, 'mass': 186.2, 'rho': 20.5, 'sym': 'Re'},
                 {'Z': 76, 'mass': 190.2, 'rho': 22.48, 'sym': 'Os'},
                 {'Z': 77, 'mass': 192.2, 'rho': 22.42, 'sym': 'Ir'},
                 {'Z': 78, 'mass': 195.09, 'rho': 21.45, 'sym': 'Pt'},
                 {'Z': 79, 'mass': 196.97, 'rho': 19.3, 'sym': 'Au'},
                 {'Z': 80, 'mass': 200.59, 'rho': 13.59, 'sym': 'Hg'},
                 {'Z': 81, 'mass': 204.37, 'rho': 11.86, 'sym': 'Tl'},
                 {'Z': 82, 'mass': 207.17, 'rho': 11.34, 'sym': 'Pb'},
                 {'Z': 83, 'mass': 208.98, 'rho': 9.8, 'sym': 'Bi'},
                 {'Z': 84, 'mass': 209.0, 'rho': 9.2, 'sym': 'Po'},
                 {'Z': 85, 'mass': 210.0, 'rho': 6.4, 'sym': 'At'},
                 {'Z': 86, 'mass': 222.0, 'rho': 4.4, 'sym': 'Rn'},
                 {'Z': 87, 'mass': 223.0, 'rho': 2.9, 'sym': 'Fr'},
                 {'Z': 88, 'mass': 226.0, 'rho': 5.0, 'sym': 'Ra'},
                 {'Z': 89, 'mass': 227.0, 'rho': 10.1, 'sym': 'Ac'},
                 {'Z': 90, 'mass': 232.04, 'rho': 11.7, 'sym': 'Th'},
                 {'Z': 91, 'mass': 231.0, 'rho': 15.4, 'sym': 'Pa'},
                 {'Z': 92, 'mass': 238.03, 'rho': 19.1, 'sym': 'U'},
                 {'Z': 93, 'mass': 237.0, 'rho': 20.2, 'sym': 'Np'},
                 {'Z': 94, 'mass': 244.0, 'rho': 19.82, 'sym': 'Pu'},
                 {'Z': 95, 'mass': 243.0, 'rho': 12.0, 'sym': 'Am'},
                 {'Z': 96, 'mass': 247.0, 'rho': 13.51, 'sym': 'Cm'},
                 {'Z': 97, 'mass': 247.0, 'rho': 14.78, 'sym': 'Bk'},
                 {'Z': 98, 'mass': 251.0, 'rho': 15.1, 'sym': 'Cf'},
                 {'Z': 99, 'mass': 252.0, 'rho': 8.84, 'sym': 'Es'},
                 {'Z': 100, 'mass': 257.0, 'rho': np.nan, 'sym': 'Fm'}]

# make an empty dictionary
OTHER_VAL = dict()
# fill it with the data keyed on the symbol
OTHER_VAL.update((elm['sym'].lower(), elm) for elm in elm_data_list)
# also add entries with it keyed on atomic number
OTHER_VAL.update((elm['Z'], elm) for elm in elm_data_list)


@functools.total_ordering
class Element(object):
    """
    Object to return all the elemental information
    related to fluorescence


    Parameters
    ----------
    element : str or int
        Element symbol or element atomic Z


    Attributes
    ----------
    name : str
    Z : int
    mass : float
    density : float
    emission_line : `XrayLibWrap`
    cs : function
    bind_energy : `XrayLibWrap`
    jump_factor : `XrayLibWrap`
    fluor_yield : `XrayLibWrap`


    Examples
    --------
    Create an `Element` object

    >>> e = Element('Zn') # or e = Element(30), 30 is atomic number

    Get the emmission energy for the Kα1 line.

    >>> e.emission_line['Ka1'] #
    8.638900756835938

    Cross section for emission line Kα1 with 10 keV incident energy

    >>> e.cs(10)['Ka1']
    54.756561279296875

    fluorescence yield for K shell

    >>> e.fluor_yield['K']
    0.46936899423599243

    atomic mass

    >>> e.mass
    65.37

    density

    >>> e.density
    7.14

    Find all emission lines within with in the range [9.5, 10.5] keV with
    an incident energy of 12 KeV.

    >>> e.find(10, 0.5, 12)
    {'kb1': 9.571999549865723}

    List all of the known emission lines

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

    List all of the known cross sections

    >>> e.cs(10).all
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

    @property
    def emission_line(self):
        """Emission line information, `XrayLibWrap`

        Emission line can be used as a unique characteristic
        for qualitative identification of the element.
        line is string type and defined as 'Ka1', 'Kb1'.
        unit in KeV
        """
        return self._emission_line

    @property
    def cs(self):
        """Fluorescence cross section function, `function`

        Returns a function of energy which returns the
        elemental cross section in cm2/g

        The signature of the function is ::

           x_section = func(enery)

        where `energy` in in keV and `x_section` is in
        cm²/g
        """
        def myfunc(incident_energy):
            return XrayLibWrap_Energy(self._z, 'cs',
                                      incident_energy)
        return myfunc

    @property
    def bind_energy(self):
        """Binding energy, `XrayLibWrap`

        Binding energy is a measure of the energy required
        to free electrons from their atomic orbits.
        shell is string type and defined as "K", "L1".
        unit in KeV
        """
        return self._bind_energy

    @property
    def jump_factor(self):
        """Jump Factor, `XrayLibWrap`

        Absorption jump factor is defined as the fraction
        of the total absorption that is associated with
        a given shell rather than for any other shell.
        shell is string type and defined as "K", "L1".
        """
        return self._jump_factor

    @property
    def fluor_yield(self):
        """fluorescence quantum yield, `XrayLibWrap`

        The fluorescence quantum yield gives the efficiency
        of the fluorescence process, and is defined as the ratio of the
        number of photons emitted to the number of photons absorbed.
        shell is string type and defined as "K", "L1".
        """
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
    """High-level interface to xraylib.

    This class exposes various functions in xraylib

    This is an interface to wrap xraylib to perform calculation related
    to xray fluorescence.

    The code does one to one map between user options,
    such as emission line, or binding energy, to xraylib function calls.

    Parameters
    ----------
    element : int
        atomic number
    info_type : {'lines',  'binding_e', 'jump', 'yield'}
        option to choose which physics quantity to calculate as follows:
        :lines: emission lines
        :binding_e: binding energy
        :jump: absorption jump factor
        :yield: fluorescence yield

    Attributes
    ----------
    info_type : str


    Examples
    --------
    Access the lines for zinc

    >>> x = XrayLibWrap(30, 'lines') # 30 is atomic number for element Zn

    Access the energy of the Kα1 line.

    >>> x['Ka1'] # energy of emission line Ka1
    8.047800064086914

    List all of the lines and their energies

    >>> x.all  # list energy of all the lines
    [(u'ka1', 8.047800064086914),
     (u'ka2', 8.027899742126465),
     (u'kb1', 8.90530014038086),
     (u'kb2', 0.0),
     (u'la1', 0.9294999837875366),
     (u'la2', 0.9294999837875366),
     (u'lb1', 0.949400007724762),
     (u'lb2', 0.0),
     (u'lb3', 1.0225000381469727),
     (u'lb4', 1.0225000381469727),
     (u'lb5', 0.0),
     (u'lg1', 0.0),
     (u'lg2', 0.0),
     (u'lg3', 0.0),
     (u'lg4', 0.0),
     (u'll', 0.8112999796867371),
     (u'ln', 0.8312000036239624),
     (u'ma1', 0.0),
     (u'ma2', 0.0),
     (u'mb', 0.0),
     (u'mg', 0.0)]
    """
    # valid options for the info_type input parameter for the init method
    opts_info_type = ['lines', 'binding_e', 'jump', 'yield']

    def __init__(self, element, info_type):
        self._element = element
        self._map, self._func = XRAYLIB_MAP[info_type]
        self._keys = sorted(list(six.iterkeys(self._map)))
        self._info_type = info_type

    @property
    def all(self):
        """List the physics quantity for all the lines or shells.
        """
        return list(six.iteritems(self))

    def __getitem__(self, key):
        """
        Call xraylib function to calculate physics quantity.  A return
        value of 0 means that the quantity not valid.

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

    @property
    def info_type(self):
        """
        option to choose which physics quantity to calculate as follows:

        """
        return self._info_type


class XrayLibWrap_Energy(XrayLibWrap):
    """
    This is an interface to wrap xraylib
    to perform calculation on fluorescence
    cross section, or other incident energy
    related quantity.

    Attributes
    ----------
    incident_energy : float
    info_type : str


    Parameters
    ----------
    element : int
        atomic number
    info_type : {'cs'}
        option to calculate physics quantities which depend on
        incident energy.  Valid values are

        :cs: cross section, unit in cm2/g

    incident_energy : float
        incident energy for fluorescence in KeV

    Examples
    --------
    Cross section of zinc with an incident X-ray at 12 KeV

    >>> x = XrayLibWrap_Energy(30, 'cs', 12)

    Compute the cross sec of the Kα1 line.

    >>> x['Ka1'] # cross section for Ka1, unit in cm2/g
    34.44424057006836
    """
    opts_info_type = ['cs']

    def __init__(self, element, info_type, incident_energy):
        super(XrayLibWrap_Energy, self).__init__(element, info_type)
        self._incident_energy = incident_energy
        self._info_type = info_type

    @property
    def incident_energy(self):
        """
        Incident x-ray energy in keV, float
        """
        return self._incident_energy

    @incident_energy.setter
    def incident_energy(self, val):
        """
        Parameters
        ----------
        val : float
            new incident x-ray energy in keV
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


def emission_line_search(line_e, delta_e,
                         incident_energy, element_list=None):
    """Find elements which have an emission line near an energy

    This function returns a dict keyed on element type of all
    elements that have an emission line with in `delta_e` of
    `line_e` at the given x-ray energy.

    Parameters
    ----------
    line_e : float
         energy value to search for in KeV
    delta_e : float
         difference compared to energy in KeV
    incident_energy : float
        incident x-ray energy in KeV
    element_list : list, optional
         List of elements to restrict search to.

         Element abbreviations can be any mix of upper and
         lower case, e.g., Hg, hG, hg, HG

    Returns
    -------
    lines_dict : dict
        element and associate emission lines

    """
    if element_list is None:
        element_list = range(1, 101)

    search_list = [Element(item) for item in element_list]

    cand_lines = [e.line_near(line_e, delta_e, incident_energy)
                  for e in search_list]

    out_dict = dict()
    for e, lines in zip(search_list, cand_lines):
        if lines:
            out_dict[e.name] = lines

    return out_dict


# http://stackoverflow.com/questions/3624753/how-to-provide-additional-initialization-for-a-subclass-of-namedtuple
class HKL(namedtuple('HKL', 'h k l')):
    '''
    Namedtuple sub-class miller indicies (HKL)

    This class enforces that the values are integers.

    Parameters
    ----------
    h : int
    k : int
    l : int

    Attributes
    ----------
    length
    h
    k
    l
    '''
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        args = [int(_) for _ in args]
        for k in list(kwargs):
            kwargs[k] = int(kwargs[k])

        return super(HKL, cls).__new__(cls, *args, **kwargs)

    @property
    def length(self):
        """
        The L2 norm (length) of the hkl vector.
        """
        return np.linalg.norm(self)


class Reflection(namedtuple('Reflection', ('d', 'hkl', 'q'))):
    """
    Namedtuple sub-class for scattering reflection information

    Parameters
    ----------
    d : float
        Plane-spacing

    HKL : `HKL`
        miller indicies

    q : float
        q-value of the reflection

    Attributes
    ----------
    d
    HKL
    q
    """
    __slots__ = ()


class PowderStandard(object):
    """
    Class for providing safe access to powder calibration standards
    data.

    Parameters
    ----------
    name : str
        Name of the standard

    reflections : list
        A list of (d, (h, k, l), q) values.
    """
    def __init__(self, name, reflections):
        self._reflections = [Reflection(d, HKL(*hkl), q)
                             for d, hkl, q in reflections]
        self._reflections.sort(key=lambda x: x[-1])
        self._name = name

    def __str__(self):
        return "Calibration standard: {}".format(self.name)

    __repr__ = __str__

    @property
    def name(self):
        """
        Name of the calibration standard
        """
        return self._name

    @property
    def reflections(self):
        """
        List of the known reflections
        """
        return self._reflections

    def __iter__(self):
        return iter(self._reflections)

    def convert_2theta(self, wavelength):
        """
        Convert the measured 2theta values to a different wavelength

        Parameters
        ----------
        wavelength : float
            The new lambda in Angstroms

        Returns
        -------
        two_theta : array
            The new 2theta values in radians
        """
        q = np.array([_.q for _ in self])
        return q_to_twotheta(q, wavelength)

    @classmethod
    def from_lambda_2theta_hkl(cls, name, wavelength, two_theta, hkl=None):
        """
        Method to construct a PowderStandard object from calibrated
        :math:`2\\theata` values.

        Parameters
        ----------
        name : str
            The name of the standard

        wavelength : float
            The wavelength that the calibration data was taken at

        two_theta : array
            The calibrated :math:`2\\theta` values

        hkl : list, optional
            List of (h, k, l) tuples of the Miller indicies that go
            with each measured :math:`2\\theta`.  If not given then
            all of the miller indicies are stored as (0, 0, 0).

        Returns
        -------
        standard : PowderStandard
            The standard object
        """
        q = twotheta_to_q(two_theta, wavelength)
        d = q_to_d(q)
        if hkl is None:
            # todo write test that hits this line
            hkl = repeat((0, 0, 0))
        return cls(name, zip(d, hkl, q))

    @classmethod
    def from_d(cls, name, d, hkl=None):
        """
        Method to construct a PowderStandard object from known
        :math:`d` values.

        Parameters
        ----------
        name : str
            The name of the standard

        d : array
            The known plane spacings

        hkl : list, optional
            List of (h, k, l) tuples of the Miller indicies that go
            with each measured :math:`2\\theta`.  If not given then
            all of the miller indicies are stored as (0, 0, 0).

        Returns
        -------
        standard : PowderStandard
            The standard object
        """
        q = d_to_q(d)
        if hkl is None:
            hkl = repeat((0, 0, 0))
        return cls(name, zip(d, hkl, q))

    def __len__(self):
        return len(self._reflections)


# Si data taken from
# https://www-s.nist.gov/srmors/certificates/640D.pdf?CFID=3219362&CFTOKEN=c031f50442c44e42-57C377F6-BC7A-395A-F39B8F6F2E4D0246&jsessionid=f030c7ded9b463332819566354567a698744

# CeO2 data taken from
# http://11bm.xray.aps.anl.gov/documents/NISTSRM/NIST_SRM_676b_%5BZnO,TiO2,Cr2O3,CeO2%5D.pdf
calibration_standards = {'Si':
                         PowderStandard.from_lambda_2theta_hkl(name='Si',
                                           wavelength=1.5405929,
                                           two_theta=np.deg2rad([
                                               28.441, 47.3,
                                               56.119, 69.126,
                                               76.371, 88.024,
                                               94.946, 106.7,
                                               114.082, 127.532,
                                               136.877]),
                                           hkl=(
                                               (1, 1, 1), (2, 2, 0),
                                               (3, 1, 1), (4, 0, 0),
                                               (3, 3, 1), (4, 2, 2),
                                               (5, 1, 1), (4, 4, 0),
                                               (5, 3, 1), (6, 2, 0),
                                               (5, 3, 3))),
                            'CeO2':
                            PowderStandard.from_lambda_2theta_hkl(name='CeO2',
                                           wavelength=1.5405929,
                                           two_theta=np.deg2rad([
                                               28.61, 33.14,
                                               47.54, 56.39,
                                               59.14, 69.46]),
                                           hkl=(
                                               (1, 1, 1), (2, 0, 0),
                                               (2, 2, 0), (3, 1, 1),
                                               (2, 2, 2), (4, 0, 0))),
                            'Al2O3':
                            PowderStandard.from_lambda_2theta_hkl(name='Al2O3',
                                           wavelength=1.5405929,
                                           two_theta=np.deg2rad([
                                               25.574, 35.149,
                                               37.773, 43.351,
                                               52.548, 57.497,
                                               66.513, 68.203,
                                               76.873, 77.233,
                                               84.348, 88.994,
                                               91.179, 95.240,
                                               101.070, 116.085,
                                               116.602, 117.835,
                                               122.019, 127.671,
                                               129.870, 131.098,
                                               136.056, 142.314,
                                               145.153, 149.185,
                                               150.102, 150.413,
                                               152.380]),
                                           hkl=(
                                               (0, 1, 2), (1, 0, 4),
                                               (1, 1, 0), (1, 1, 3),
                                               (0, 2, 4), (1, 1, 6),
                                               (2, 1, 4), (3, 0, 0),
                                               (1, 0, 10), (1, 1, 9),
                                               (2, 2, 3), (0, 2, 10),
                                               (1, 3, 4), (2, 2, 6),
                                               (2, 1, 10), (3, 2, 4),
                                               (0, 1, 14), (4, 1, 0),
                                               (4, 1, 3), (1, 3, 10),
                                               (3, 0, 12), (2, 0, 14),
                                               (1, 4, 6), (1, 1, 15),
                                               (4, 0, 10), (0, 5, 4),
                                               (1, 2, 14), (1, 0, 16),
                                               (3, 3, 0))),
                            'LaB6':
                            PowderStandard.from_d(name='LaB6',
                                                  d=[4.156,
                                                     2.939,
                                                     2.399,
                                                     2.078,
                                                     1.859,
                                                     1.697,
                                                     1.469,
                                                     1.385,
                                                     1.314,
                                                     1.253,
                                                     1.200,
                                                     1.153,
                                                     1.111,
                                                     1.039,
                                                     1.008,
                                                     0.980,
                                                     0.953,
                                                     0.929,
                                                     0.907,
                                                     0.886,
                                                     0.848,
                                                     0.831,
                                                     0.815,
                                                     0.800]),
                            'Ni':
                             PowderStandard.from_d(name='Ni',
                                                   d=[2.03458234862,
                                                     1.762,
                                                     1.24592214845,
                                                     1.06252597829,
                                                     1.01729117431,
                                                     0.881,
                                                     0.80846104616,
                                                     0.787990355271,
                                                     0.719333487797,
                                                     0.678194116208,
                                                     0.622961074225,
                                                     0.595664718733,
                                                     0.587333333333,
                                                     0.557193323722,
                                                     0.537404961852,
                                                     0.531262989146,
                                                     0.508645587156,
                                                     0.493458701611,
                                                     0.488690872874,
                                                     0.47091430825,
                                                     0.458785722296,
                                                     0.4405,
                                                     0.430525121912,
                                                     0.427347771314])}
"""
Calibration standards

A dictionary holding known powder-pattern calibration standards
"""
