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

"""
Module for xray scattering
"""
from __future__ import absolute_import, division, print_function

import logging
from collections import namedtuple
from itertools import repeat

import numpy as np

from ..utils import d_to_q, q_to_d, q_to_twotheta, twotheta_to_q

logger = logging.getLogger(__name__)


# http://stackoverflow.com/questions/3624753/how-to-provide-additional-initialization-for-a-subclass-of-namedtuple
class HKL(namedtuple("HKL", "h k l")):
    """
    Namedtuple sub-class miller indicies (HKL)

    This class enforces that the values are integers.

    Parameters
    ----------
    h : int
    k : int
    l : int

    """

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


class Reflection(namedtuple("Reflection", ("d", "hkl", "q"))):
    """
    Namedtuple sub-class for scattering reflection information

    Parameters
    ----------
    d : float
        Plane-spacing

    hkl : `hkl`
        miller indicies

    q : float
        q-value of the reflection

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
        self._reflections = [Reflection(d, HKL(*hkl), q) for d, hkl, q in reflections]
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
        Convert the measured $2theta$ values to a different wavelength

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


"""
Calibration standards

A dictionary holding known powder-pattern calibration standards
"""
# Si (Standard Reference Material 640d) data taken from
# https://www-s.nist.gov/srmors/certificates/640D.pdf?CFID=3219362&CFTOKEN=c031f50442c44e42-57C377F6-BC7A-395A-F39B8F6F2E4D0246&jsessionid=f030c7ded9b463332819566354567a698744

# CeO2 (Standard Reference Material 674b) data taken from
# http://11bm.xray.aps.anl.gov/documents/NISTSRM/NIST_SRM_676b_%5BZnO,TiO2,Cr2O3,CeO2%5D.pdf

# Alumina (Al2O3), (Standard Reference Material 676a) taken from
# https://www-s.nist.gov/srmors/certificates/676a.pdf?CFID=3259108&CFTOKEN=fa5bb0075f99948c-FA6ABBDA-9691-7A6B-FBE24BE35748DC08&jsessionid=f030e1751fc5365cac74417053f2c344f675
#: Mapping of known calibration standards and their reflections
calibration_standards = {
    "Si": PowderStandard.from_lambda_2theta_hkl(
        name="Si",
        wavelength=1.5405929,
        two_theta=np.deg2rad(
            [28.441, 47.3, 56.119, 69.126, 76.371, 88.024, 94.946, 106.7, 114.082, 127.532, 136.877]
        ),
        hkl=(
            (1, 1, 1),
            (2, 2, 0),
            (3, 1, 1),
            (4, 0, 0),
            (3, 3, 1),
            (4, 2, 2),
            (5, 1, 1),
            (4, 4, 0),
            (5, 3, 1),
            (6, 2, 0),
            (5, 3, 3),
        ),
    ),
    "CeO2": PowderStandard.from_lambda_2theta_hkl(
        name="CeO2",
        wavelength=1.5405929,
        two_theta=np.deg2rad([28.61, 33.14, 47.54, 56.39, 59.14, 69.46]),
        hkl=((1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1), (2, 2, 2), (4, 0, 0)),
    ),
    "Al2O3": PowderStandard.from_lambda_2theta_hkl(
        name="Al2O3",
        wavelength=1.5405929,
        two_theta=np.deg2rad(
            [
                25.574,
                35.149,
                37.773,
                43.351,
                52.548,
                57.497,
                66.513,
                68.203,
                76.873,
                77.233,
                84.348,
                88.994,
                91.179,
                95.240,
                101.070,
                116.085,
                116.602,
                117.835,
                122.019,
                127.671,
                129.870,
                131.098,
                136.056,
                142.314,
                145.153,
                149.185,
                150.102,
                150.413,
                152.380,
            ]
        ),
        hkl=(
            (0, 1, 2),
            (1, 0, 4),
            (1, 1, 0),
            (1, 1, 3),
            (0, 2, 4),
            (1, 1, 6),
            (2, 1, 4),
            (3, 0, 0),
            (1, 0, 10),
            (1, 1, 9),
            (2, 2, 3),
            (0, 2, 10),
            (1, 3, 4),
            (2, 2, 6),
            (2, 1, 10),
            (3, 2, 4),
            (0, 1, 14),
            (4, 1, 0),
            (4, 1, 3),
            (1, 3, 10),
            (3, 0, 12),
            (2, 0, 14),
            (1, 4, 6),
            (1, 1, 15),
            (4, 0, 10),
            (0, 5, 4),
            (1, 2, 14),
            (1, 0, 16),
            (3, 3, 0),
        ),
    ),
    "LaB6": PowderStandard.from_d(
        name="LaB6",
        d=[
            4.156,
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
            0.800,
        ],
    ),
    "Ni": PowderStandard.from_d(
        name="Ni",
        d=[
            2.03458234862,
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
            0.427347771314,
        ],
    ),
}
