# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# @author: Li Li (lili@bnl.gov)                                        #
# created on 07/10/2014                                                #
#                                                                      #
# Original code:                                                       #
# @author: Mirna Lerotic, 2nd Look Consulting                          #
#         http://www.2ndlookconsulting.com/                            #
# Copyright (c) 2013, Stefan Vogt, Argonne National Laboratory         #
# All rights reserved.                                                 #
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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import sys

from lmfit.models import (ConstantModel, LinearModel, QuadraticModel,
                          ParabolicModel, PolynomialModel, VoigtModel,
                          PseudoVoigtModel, Pearson7Model, StudentsTModel,
                          BreitWignerModel, GaussianModel, LorentzianModel,
                          LognormalModel, DampedOscillatorModel,
                          ExponentialGaussianModel, SkewedGaussianModel,
                          DonaichModel, PowerLawModel, ExponentialModel,
                          StepModel, RectangleModel, ExpressionModel)

from .models import (Lorentzian2Model, ComptonModel, ElasticModel)

from lmfit.lineshapes import (pearson7, breit_wigner, damped_oscillator,
                              logistic, lognormal, students_t, expgaussian,
                              donaich, skewed_gaussian, skewed_voigt, step,
                              rectangle, exponential, powerlaw, linear,
                              parabolic)
from .lineshapes import (gaussian, lorentzian, lorentzian2, voigt, pvoigt,
                         gaussian_tail, gausssian_step, elastic, compton)

# construct lists of the models that can be used
model_list = [
    ConstantModel, LinearModel, QuadraticModel, ParabolicModel,
    PolynomialModel, GaussianModel, LorentzianModel, VoigtModel,
    PseudoVoigtModel, Pearson7Model, StudentsTModel, BreitWignerModel,
    LognormalModel, DampedOscillatorModel, ExponentialGaussianModel,
    SkewedGaussianModel, DonaichModel, PowerLawModel, ExponentialModel,
    StepModel, RectangleModel, Lorentzian2Model, ComptonModel, ElasticModel
].sort(key=lambda s: str(s).split('.')[-1])

# construct a list of the models that can be used
lineshapes_list = [
    gaussian, lorentzian, voigt, pvoigt, pearson7, breit_wigner,
    damped_oscillator, logistic, lognormal, students_t, expgaussian, donaich,
    skewed_gaussian, skewed_voigt, step, rectangle, exponential, powerlaw,
    linear, parabolic, lorentzian2, compton, elastic, gausssian_step,
    gaussian_tail
].sort(key = lambda s: str(s))
