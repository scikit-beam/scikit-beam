# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# @author: Li Li (lili@bnl.gov)                                        #
# created on 07/16/2014                                                #
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
from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.signal

_defaults = {"con_val_no_bin": 3, "con_val_bin": 5, "iter_num_no_bin": 3, "iter_num_bin": 5}


def snip_method(
    spectrum,
    e_off,
    e_lin,
    e_quad,
    xmin=0,
    xmax=4096,
    epsilon=2.96,
    width=0.5,
    decrease_factor=np.sqrt(2),
    spectral_binning=None,
    con_val=None,
    iter_num=None,
    width_threshold=0.5,
):
    """
    use snip algorithm to obtain background

    Parameters
    ----------
    spectrum : array
        intensity spectrum
    e_off : float
        energy calibration, such as e_off + e_lin * energy + e_quad * energy^2
    e_lin : float
        energy calibration, such as e_off + e_lin * energy + e_quad * energy^2
    e_quad : float
        energy calibration, such as e_off + e_lin * energy + e_quad * energy^2
    xmin : float, optional
        smallest index to define the range
    xmax : float, optional
        largest index to define the range
    epsilon : float, optional
        energy to create a hole-electron pair
        for Ge 2.96, for Si 3.61 at 300K
        needs to double check this value
    width : int, optional
        window size to adjust how much to shift background
    decrease_factor : float, optional
        gradually decrease of window size, default as sqrt(2)
    spectral_binning : float, optional
        bin the data into different size
    con_val : int, optional
        size of scipy.signal.windows.boxcar to convolve the spectrum.

        Default value is controlled by the keys `con_val_no_bin`
        and `con_val_bin` in the defaults dictionary, depending
        on if spectral_binning is used or not

    iter_num : int, optional
        Number of iterations.

        Default value is controlled by the keys `iter_num_no_bin`
        and `iter_num_bin` in the defaults dictionary, depending
        on if spectral_binning is used or not

    width_threshold : float, optional
        stop point of the algorithm

    Returns
    -------
    background : array
        output results with peak removed

    References
    ----------

    .. [1] C.G. Ryan etc, "SNIP, a statistics-sensitive background
           treatment for the quantitative analysis of PIXE spectra in
           geoscience applications", Nuclear Instruments and Methods in
           Physics Research Section B, vol. 34, 1998.
    """
    # clean input a bit
    if con_val is None:
        if spectral_binning is None:
            con_val = _defaults["con_val_no_bin"]
        else:
            con_val = _defaults["con_val_bin"]

    if iter_num is None:
        if spectral_binning is None:
            iter_num = _defaults["iter_num_no_bin"]
        else:
            iter_num = _defaults["iter_num_bin"]

    background = np.array(spectrum)
    n_background = background.size

    energy = np.arange(n_background, dtype=np.float64)

    if spectral_binning is not None:
        energy = energy * spectral_binning

    energy = e_off + energy * e_lin + energy**2 * e_quad

    # transfer from std to fwhm
    std_fwhm = 2 * np.sqrt(2 * np.log(2))
    tmp = (e_off / std_fwhm) ** 2 + energy * epsilon * e_lin
    tmp[tmp < 0] = 0
    fwhm = std_fwhm * np.sqrt(tmp)

    # smooth the background
    s = scipy.signal.windows.boxcar(con_val)

    # For background remove, we only care about the central parts
    # where there are peaks. On the boundary part, we don't care
    # the accuracy so much. But we need to pay attention to edge
    # effects in general convolution.
    A = s.sum()
    background = scipy.signal.convolve(background, s, mode="same") / A

    window_p = width * fwhm / e_lin
    if spectral_binning is not None and spectral_binning > 0:
        window_p = window_p / 2.0

    background = np.log(np.log(background + 1) + 1)

    index = np.arange(n_background)

    # FIRST SNIPPING
    for j in range(iter_num):
        lo_index = np.clip(index - window_p, np.max([xmin, 0]), np.min([xmax, n_background - 1]))
        hi_index = np.clip(index + window_p, np.max([xmin, 0]), np.min([xmax, n_background - 1]))

        temp = (background[lo_index.astype(np.int64)] + background[hi_index.astype(np.int64)]) / 2.0

        bg_index = background > temp
        background[bg_index] = temp[bg_index]

    current_width = window_p
    max_current_width = np.amax(current_width)

    while max_current_width >= width_threshold:
        lo_index = np.clip(index - current_width, np.max([xmin, 0]), np.min([xmax, n_background - 1]))
        hi_index = np.clip(index + current_width, np.max([xmin, 0]), np.min([xmax, n_background - 1]))

        temp = (background[lo_index.astype(np.int64)] + background[hi_index.astype(np.int64)]) / 2.0

        bg_index = background > temp
        background[bg_index] = temp[bg_index]

        # decrease the width and repeat
        current_width = current_width / decrease_factor
        max_current_width = np.amax(current_width)

    background = np.exp(np.exp(background) - 1) - 1

    inf_ind = np.where(~np.isfinite(background))
    background[inf_ind] = 0.0

    return background
