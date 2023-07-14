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

import logging
from collections.abc import Mapping

import numpy as np
import six
from packaging import version

from ..constants.basic import BasicElement, doc_attrs, doc_ex, doc_params
from ..utils import NotInstalledError, verbosedict

logger = logging.getLogger(__name__)

line_name = [
    "Ka1",
    "Ka2",
    "Ka3",
    "Kb1",
    "Kb2",
    "Kb3",
    "Kb4",
    "kb5",
    "La1",
    "La2",
    "Lb1",
    "Lb2",
    "Lb3",
    "Lb4",
    "Lb5",
    "Lg1",
    "Lg2",
    "Lg3",
    "Lg4",
    "Ll",
    "Ln",
    "Ma1",
    "Ma2",
    "Mb",
    "Mg",
]

bindingE = [
    "K",
    "L1",
    "L2",
    "L3",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "N1",
    "N2",
    "N3",
    "N4",
    "N5",
    "N6",
    "N7",
    "O1",
    "O2",
    "O3",
    "O4",
    "O5",
    "P1",
    "P2",
    "P3",
]


class XraylibNotInstalledError(NotInstalledError):
    message_post = (
        "xraylib is not installed. Please see "
        "https://github.com/tschoonj/xraylib "
        "or https://binstar.org/tacaswell/xraylib "
        "for help on installing xraylib"
    )

    def __init__(self, caller, *args, **kwargs):
        message = "The call to {} cannot be completed because {}" "".format(caller, self.message_post)
        super(XraylibNotInstalledError, self).__init__(message, *args, **kwargs)


try:
    import xraylib
except ImportError:
    logger.warning("Xraylib is not installed on your machine. " + XraylibNotInstalledError.message_post)
    xraylib = None

if xraylib is None:
    # do nothing, for now
    pass
else:
    xraylib.XRayInit()
    if version.parse(xraylib.__version__) < version.parse("4.0.0"):
        xraylib.SetErrorMessages(0)

    line_list = [
        xraylib.KA1_LINE,
        xraylib.KA2_LINE,
        xraylib.KA3_LINE,
        xraylib.KB1_LINE,
        xraylib.KB2_LINE,
        xraylib.KB3_LINE,
        xraylib.KB4_LINE,
        xraylib.KB5_LINE,
        xraylib.LA1_LINE,
        xraylib.LA2_LINE,
        xraylib.LB1_LINE,
        xraylib.LB2_LINE,
        xraylib.LB3_LINE,
        xraylib.LB4_LINE,
        xraylib.LB5_LINE,
        xraylib.LG1_LINE,
        xraylib.LG2_LINE,
        xraylib.LG3_LINE,
        xraylib.LG4_LINE,
        xraylib.LL_LINE,
        xraylib.LE_LINE,
        xraylib.MA1_LINE,
        xraylib.MA2_LINE,
        xraylib.MB_LINE,
        xraylib.MG_LINE,
    ]

    shell_list = [
        xraylib.K_SHELL,
        xraylib.L1_SHELL,
        xraylib.L2_SHELL,
        xraylib.L3_SHELL,
        xraylib.M1_SHELL,
        xraylib.M2_SHELL,
        xraylib.M3_SHELL,
        xraylib.M4_SHELL,
        xraylib.M5_SHELL,
        xraylib.N1_SHELL,
        xraylib.N2_SHELL,
        xraylib.N3_SHELL,
        xraylib.N4_SHELL,
        xraylib.N5_SHELL,
        xraylib.N6_SHELL,
        xraylib.N7_SHELL,
        xraylib.O1_SHELL,
        xraylib.O2_SHELL,
        xraylib.O3_SHELL,
        xraylib.O4_SHELL,
        xraylib.O5_SHELL,
        xraylib.P1_SHELL,
        xraylib.P2_SHELL,
        xraylib.P3_SHELL,
    ]

    line_dict = verbosedict((k.lower(), v) for k, v in zip(line_name, line_list))

    shell_dict = verbosedict((k.lower(), v) for k, v in zip(bindingE, shell_list))

    XRAYLIB_MAP = verbosedict(
        {
            "lines": (line_dict, xraylib.LineEnergy),
            "cs": (line_dict, xraylib.CS_FluorLine_Kissel),
            "csb": (line_dict, xraylib.CSb_FluorLine_Kissel),
            "binding_e": (shell_dict, xraylib.EdgeEnergy),
            "jump": (shell_dict, xraylib.JumpFactor),
            "yield": (shell_dict, xraylib.FluorYield),
        }
    )


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
    opts_info_type = ["lines", "binding_e", "jump", "yield"]

    def __init__(self, element, info_type, energy=None):
        if xraylib is None:
            raise XraylibNotInstalledError(self.__class__)
        self._element = element
        self._map, self._func = XRAYLIB_MAP[info_type]
        self._keys = sorted(list(six.iterkeys(self._map)))
        self._info_type = info_type

    @property
    def all(self):
        """List the physics quantity for all the lines or shells."""
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

        try:
            # xraylib functions for xraylib < 4.0 used to return 0 in case of non-existent lines
            #   This is extensively used in scikit-beam/pyxrf to determine if the lines exist.
            #   Starting from v4.0, xraylib is raising 'ValueError' exception instead. We are
            #   imitating behavior of the old xraylib by catching the exception and returning 0.
            val = self._func(self._element, self._map[key.lower()])
        except ValueError:
            val = 0
        return val

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

    Parameters
    ----------
    element : int
        atomic number
    info_type : {'cs', 'csb'}, optional
        option to calculate physics quantities which depend on
        incident energy.
        See Class attribute `opts_info_type` for valid options

        :cs: cross section, unit in cm2/g
        :csb: cross section, unit in barns/atom

    incident_energy : float
        incident energy for fluorescence in KeV

    Examples
    --------
    >>> # Cross section of zinc with an incident X-ray at 12 KeV
    >>> x = XrayLibWrap_Energy(30, 'cs', 12)
    >>> # Compute the cross section of the Kα1 line.
    >>> x['Ka1'] # cross section for Ka1, unit in cm2/g
    34.68250086875594
    >>> xb = XrayLibWrap_Energy(30, 'csb', 12)
    >>> # Compute the cross section of the Kα1 line.
    >>> xb['Ka1'] # cross section for Ka1, unit in barns/atom
    3765.3415913117224
    """

    opts_info_type = ["cs", "csb"]

    def __init__(self, element, info_type, incident_energy):
        if xraylib is None:
            raise XraylibNotInstalledError(self.__class__)

        super(XrayLibWrap_Energy, self).__init__(element, info_type)
        self._incident_energy = incident_energy

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
        self._incident_energy = float(val)

    def __getitem__(self, key):
        """
        Call xraylib function to calculate physics quantity.

        Parameters
        ----------
        key : str
            defines which physics quantity to calculate
        """
        try:
            # xraylib functions for xraylib < 4.0 used to return 0 in case of non-existent lines
            #   This is extensively used in scikit-beam/pyxrf to determine if the lines exist.
            #   Starting from v4.0, xraylib is raising 'ValueError' exception instead. We are
            #   imitating behavior of the old xraylib by catching the exception and returning 0.
            val = self._func(self._element, self._map[key.lower()], self._incident_energy)
        except ValueError:
            val = 0

        return val


# redefine the doc_title for xrf elements
doc_title = """
    Object to return all the elemental information related to fluorescence
    """
# dont change the doc_params
doc_params = doc_params
#
doc_attrs += """    emission_line : `XrayLibWrap`
    cs : function
    csb : function
    bind_energy : `XrayLibWrap`
    jump_factor : `XrayLibWrap`
    fluor_yield : `XrayLibWrap`
    """
doc_ex += (
    """
    >>> from skbeam.core.constants.xrf import XrfElement as Element
    >>> e = Element('Zn')
    >>> # Get the emission energy for the Kα1 line.
    >>> e.emission_line['Ka1']
    8.638900756835938

    >>> # Cross section [barns/atom] for Kα1 line at 10 keV incident energy
    >>> e.csb(10)['Ka1']
    5987.081587605121

    >>> # Cross section [cm2/g] for Kα1 line at 10 keV incident energy
    >>> e.cs(10)['Ka1']
    55.146912259583296

    >>> # fluorescence yield for K shell
    >>> e.fluor_yield['K']
    0.46936899423599243

    >>> # Find all emission lines within with in the range [9.5, 10.5]
    >>> # keV with an incident energy of 12 KeV.
    >>> e.find(10, 0.5, 12)
    {'kb1': 9.571999549865723}

    >>> # List all of the known emission lines
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

    >>> # List all of the known cross sections [barns/atom]
    >>> e.csb(10).all
    [('ka1', 5987.081587605121),
     ('ka2', 3076.4914784265347),
     ('kb1', 821.0572112842519),
     ('kb2', 0.0),
     ('la1', 188.06856034970164),
     ('la2', 21.213083101234524),
     ('lb1', 94.10717616654374),
     ('lb2', 0.0),
     ('lb3', 6.2207984090565),
     ('lb4', 3.3964315566384187),
     ('lb5', 0.0),
     ('lg1', 0.0),
     ('lg2', 0.0),
     ('lg3', 0.0),
     ('lg4', 0.0),
     ('ll', 11.809765990232954),
     ('ln', 4.8441078404731766),
     ('ma1', 0.0),
     ('ma2', 0.0),
     ('mb', 0.0),
     ('mg', 0.0)]
    """
    ""
)


class XrfElement(BasicElement):
    # define the docs
    __doc__ = """{}
    Parameters
    ----------{}

    Examples
    --------{}
    """.format(
        doc_title, doc_params, doc_ex
    )

    def __init__(self, element):
        if xraylib is None:
            raise XraylibNotInstalledError(self.__class__)

        super(XrfElement, self).__init__(element)

        self._emission_line = XrayLibWrap(self.Z, "lines")
        self._bind_energy = XrayLibWrap(self.Z, "binding_e")
        self._jump_factor = XrayLibWrap(self.Z, "jump")
        self._fluor_yield = XrayLibWrap(self.Z, "yield")

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
        cm2/g
        """

        def myfunc(incident_energy):
            return XrayLibWrap_Energy(self.Z, "cs", incident_energy)

        return myfunc

    @property
    def csb(self):
        """Fluorescence cross section function, `function`

        Returns a function of energy which returns the
        elemental cross section in barns/atom

        The signature of the function is ::

           x_section = func(enery)

        where `energy` in in keV and `x_section` is in
        barns/atom
        """

        def myfunc(incident_energy):
            return XrayLibWrap_Energy(self.Z, "csb", incident_energy)

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

    def line_near(self, energy, delta_e, incident_energy):
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


def emission_line_search(line_e, delta_e, incident_energy, element_list=None):
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
         List of elements to restrict search to. If no list is present,
         search on all elements.
         Element abbreviations can be any mix of upper and
         lower case, e.g., Hg, hG, hg, HG

    Returns
    -------
    lines_dict : dict
        element and associate emission lines

    """
    if xraylib is None:
        raise XraylibNotInstalledError(__name__)

    if element_list is None:
        element_list = range(1, 101)

    search_list = [XrfElement(item) for item in element_list]

    cand_lines = [e.line_near(line_e, delta_e, incident_energy) for e in search_list]

    out_dict = dict()
    for e, lines in zip(search_list, cand_lines):
        if lines:
            out_dict[e.sym] = lines

    return out_dict
