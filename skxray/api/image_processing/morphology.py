# ######################################################################
# Copyright (c) 2015, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# @author: Gabriel Iltis (giltis@bnl.gov)
#
# created on 06/24/2015                                                #
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
This module creates a namespace for morphological functions used in
the analysis of Full-Field Imaging and Image Processing
"""

import logging
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------
#Image processing: Image morphology
#-----------------------------------------------------------------------------
from scipy.ndimage.morphology import (
    binary_opening, binary_closing, binary_erosion, binary_dilation,
    grey_opening, grey_closing, grey_erosion, grey_dilation,
    binary_fill_holes, binary_propagation
)

# from skxray.image_processing.morphology import (
#     extract_material, extract_all_else
# )

from skimage.restoration import (
    denoise_bilateral, denoise_tv_bregman, denoise_tv_chambolle,
    nl_means_denoising, richardson_lucy, unsupervised_wiener, unwrap_phase,
    wiener,
)

from skimage.feature import (
    blob_dog, blob_doh, blob_log, canny, peak_local_max,
)

from skimage.morphology import (
    ball, black_tophat, convex_hull_image, convex_hull_object, medial_axis,
    reconstruction, skeletonize, watershed, white_tophat
)

__all__ = [
    # image morphology
    'binary_opening', 'binary_closing', 'binary_erosion', 'binary_dilation',
    'grey_opening', 'grey_closing', 'grey_erosion', 'grey_dilation',
    'binary_fill_holes', 'binary_propagation', 'denoise_bilateral',
    'denoise_tv_bregman', 'denoise_tv_chambolle', 'nl_means_denoising',
    'richardson_lucy', 'unsupervised_wiener', 'unwrap_phase',
    'wiener', 'blob_dog', 'blob_doh', 'blob_log', 'canny', 'peak_local_max',
    'ball', 'black_tophat', 'convex_hull_image', 'convex_hull_object',
    'medial_axis', 'reconstruction', 'skeletonize', 'watershed', 'white_tophat'
]
