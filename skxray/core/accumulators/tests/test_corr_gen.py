# # Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
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
from __future__ import absolute_import, division, print_function
import numpy as np
from numpy.testing import assert_array_almost_equal
from skxray.core.accumulators import corr_gen


def test_partial_data_correlation():
    batch_size = 100
    size = 50
    img_stack = np.random.randint(1, 10, (batch_size, size, size))

    num_levels = 2
    num_bufs = 4
    labels = np.zeros((size, size), dtype=np.int64)
    labels[2:10, 5:15] = 1

    # make sure it works with a generator
    img_gen = (img for img in img_stack)
    res1, = corr_gen.multi_tau_auto_corr_partial_data(
        num_levels, num_bufs, labels, img_gen)
    # make sure we are basically at 1
    assert np.average(res1[0][1:] - 1) < 0.01

    # compute correlation for the first half
    img_gen = (img for img in img_stack[:batch_size // 2])
    res2, = corr_gen.multi_tau_auto_corr_partial_data(
        num_levels, num_bufs, labels, img_gen)
    res3, = corr_gen.multi_tau_auto_corr_partial_data(
        num_levels, num_bufs, labels, img_stack[batch_size // 2:])

    assert_array_almost_equal(res1[0], res3[0], decimal=2)
