# ######################################################################
# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel C. Iltis, Oct. 2014
#
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
This module provides test functions for image histogram evaluation
and modification.

This tool testing set focuses on tools provided in modules that can be
accessed using:
    nsls2.img_proc.histops
    nsls2.img_proc.synth_drawing
"""
import numpy as np
import nsls2.img_proc.histops as histops
import nsls2.img_proc.synth_drawing as draw
from nsls2.core import bin_1D
from nsls2.core import bin_edges_to_centers
from numpy.testing import assert_array_almost_equal

test_array1 = np.array([[[0,0,0],[0,1,0],[0,0,0]],
                       [[0,2,0],[2,3,2],[0,2,0]],
                       [[4,4,4],[4,5,4],[4,4,4]]])

test_array2 = np.array([[[0,2,3],[4,5,6],[7,8,9]],
                       [[0,2,0],[2,3,2],[3,2,3]],
                       [[5,5,5],[4,5,4],[4,4,4]]])

def test_hist_make_from_synth_hist():
    h1 = 500
    cen1 = 0
    w1 = 400
    num_values = 3000
    x_min1 = -2400
    x_max1 = 2400

    h2 = 350
    cen2 = 4500
    w2 = 800
    x_min2 = 0
    x_max2 = 9000

    x_crv_1, y_crv_1 = draw.draw_gauss_crv(h1,
                                      cen1,
                                      w1,
                                      num_values,
                                      x_min1,
                                      x_max1)
    x_crv_2, y_crv_2 = draw.draw_gauss_crv(h2,
                                      cen2,
                                      w2,
                                      num_values,
                                      x_min2,
                                      x_max2)
    merged_crvs = np.array(zip(np.concatenate((x_crv_1, x_crv_2)),
                               np.concatenate((y_crv_1, y_crv_2))))
    hist_src_synth = merged_crvs[merged_crvs[:,0].argsort()]
    nx = (np.amax((x_min1, x_max1, x_min2, x_max2)) -
          np.amin((x_min1, x_max1, x_min2, x_max2)))
    bin_edges, vals, count = bin_1D(hist_src_synth[:,0],
                                    hist_src_synth[:,1], nx=nx)
    vals = np.floor(vals)
    bin_avg = bin_edges_to_centers(bin_edges)
    cell_location = 0
    vals_sum = 0
    synth_data=np.empty(vals.sum())
    for _ in np.arange(len(vals)):
        for counts in np.arange(vals[_]):
            synth_data[cell_location] = bin_avg[_]
            cell_location+=1

    hist, edges, avg = histops.hist_make(synth_data, len(vals))
    source_curve = np.array(zip(bin_avg, vals))
    hist_make_results = np.array(zip(avg, hist))
    area_s_curve = source_curve[:,1].sum()
    area_hist_make = hist_make_results[:,1].sum()
    assert_array_almost_equal(area_s_curve,
                              area_hist_make,
                              decimal=3,
                              err_msg=('hist_make results and synthetic curve'\
                                       'do not match. Area under synthetic '\
                                       'curve equals {0}, area under the '\
                                       'histogram generated using hist_make '\
                                       'equals: {1}'.format(area_s_curve,
                                                            area_hist_make,)))


def test_hist_make_from_synth_img():
    pass
