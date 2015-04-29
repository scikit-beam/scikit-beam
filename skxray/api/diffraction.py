#! encoding: utf-8
# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
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
This module creates a namespace for X-Ray Diffraction
"""

import logging
logger = logging.getLogger(__name__)

# import fitting models
from ..fitting.api import (
    ConstantModel, LinearModel, QuadraticModel, ParabolicModel,
    PolynomialModel, VoigtModel, PseudoVoigtModel, Pearson7Model,
    StudentsTModel, BreitWignerModel, GaussianModel, LorentzianModel,
    LognormalModel, DampedOscillatorModel, ExponentialGaussianModel,
    SkewedGaussianModel, DonaichModel, PowerLawModel, ExponentialModel,
    StepModel, RectangleModel, Lorentzian2Model, ComptonModel, ElasticModel
)

from ..recip import process_to_q, hkl_to_q

from ..constants.api import (
    BasicElement, calibration_standards
)

from ..core import (
    bin_1D, bin_edges, bin_edges_to_centers, grid3d,
    q_to_d, d_to_q,
    q_to_twotheta, twotheta_to_q,
    angle_grid, radial_grid,
)

from ..calibration import (
    refine_center, estimate_d_blind,
)

__all__ = [
    # fitting api
    'ConstantModel', 'LinearModel', 'QuadraticModel', 'ParabolicModel',
    'PolynomialModel', 'VoigtModel', 'PseudoVoigtModel', 'Pearson7Model',
    'StudentsTModel', 'BreitWignerModel', 'GaussianModel', 'LorentzianModel',
    'LognormalModel', 'DampedOscillatorModel', 'ExponentialGaussianModel',
    'SkewedGaussianModel', 'DonaichModel', 'PowerLawModel', 'ExponentialModel',
    'StepModel', 'RectangleModel', 'Lorentzian2Model', 'ComptonModel',
    'ElasticModel',

    # recip
    'process_to_q', 'hkl_to_q',

    # constants api
    'BasicElement', 'calibration_standards',

    # core
    'bin_1D', 'bin_edges', 'bin_edges_to_centers', 'grid3d', 'q_to_d',
    'd_to_q', 'q_to_twotheta', 'twotheta_to_q', 'angle_grid',
    'radial_grid',

    # calibration
    'refine_center', 'estimate_d_blind',
]
