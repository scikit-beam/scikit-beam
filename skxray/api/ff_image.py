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
"""
This module creates a namespace for Full-Field Imaging and Image Processing
"""


import logging
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------
#Image processing: Filter operations
#-----------------------------------------------------------------------------
from scipy.ndimage.filters import (gaussian_filter, median_filter,
                                   minimum_filter, maximum_filter,
                                   gaussian_gradient_magnitude,
                                   gaussian_laplace, laplace,
                                   percentile_filter, sobel, prewitt,
                                   rank_filter)


#-----------------------------------------------------------------------------
#Image processing: Histogram analysis and manipulation
#-----------------------------------------------------------------------------
from ..img_proc.histops import (hist_make, rescale_intensity_values)


#-----------------------------------------------------------------------------
#Image processing: Image morphology
#-----------------------------------------------------------------------------
from scipy.ndimage.morphology import (binary_opening, binary_closing,
                                      binary_erosion, binary_dilation,
                                      grey_opening, grey_closing,
                                      grey_erosion, grey_dilation,
                                      binary_fill_holes, binary_propagation)


#-----------------------------------------------------------------------------
#Image processing: Image thresholding
#-----------------------------------------------------------------------------
from ..img_proc.threshops import (thresh_globalGT, thresh_globalLT,
                                  thresh_bounded, thresh_adapt, thresh_otsu,
                                  thresh_yen, thresh_isodata)

thresh_adapt.k_shape = ['2D', '3D']
thresh_adapt.filter_type = ['generic', 'gaussian', 'mean', 'median']


#-----------------------------------------------------------------------------
#Image processing: Image transformation
#-----------------------------------------------------------------------------
from ..img_proc.transform import (swap_axes, flip_axis, crop_volume,
                                  rotate_volume)

swap_axes.select_axes = ['XY', 'YZ', 'XZ']
flip_axis.flip_direction = ['Flip Z', 'Flip Y', 'Flip X']
rotate_volume.rotate_axis = ['Z-axis', 'Y-axis', 'X-axis']
