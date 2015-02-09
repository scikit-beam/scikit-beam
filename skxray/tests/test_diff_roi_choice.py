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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
import logging
logger = logging.getLogger(__name__)
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)
import sys

from nose.tools import assert_equal, assert_true, raises

import skxray.diff_roi_choice as diff_roi

from skxray.testing.decorators import known_fail_if
import numpy.testing as npt


def test_roi_rectangles():
    detector_size = (15, 10)
    num_rois = 3
    roi_data = np.array(([2, 2, 3, 3], [6, 7, 3, 2], [11, 8, 5, 2]),
                        dtype=np.int64)

    xy_inds, num_pixels, pixel_list = diff_roi.roi_rectangles(num_rois,
                                                              roi_data,
                                                              detector_size)

    xy_inds_m = ([1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3,
                 3, 3, 3, 3, 3])

    num_pixels_m = [9, 6, 8]
    pixel_list_m = ([22, 23, 24, 32, 33, 34, 42, 43, 44, 67, 68, 77, 78,
                     87, 88, 118, 119, 128, 129, 138, 139, 148, 149],)

    assert_array_equal(num_pixels, num_pixels_m)
    assert_array_equal(xy_inds, np.ravel(xy_inds_m))
    assert_array_equal(pixel_list, pixel_list_m)


def test_roi_rings():
    calibrated_center = (4., 4.)
    img_dim = (20, 25)
    first_q = 2.5
    delta_q = 2
    num_qs = 10  # number of Q rings

    (q_inds, q_ring_val, num_pixels,
     pixel_list) = diff_roi.roi_rings(img_dim, calibrated_center, num_qs,
                                      first_q, delta_q)

    q_inds_m = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6,
                         6, 7, 7, 8, 8, 9, 9, 10, 1, 1, 1, 2, 2, 3, 3, 4, 4,
                         5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 1, 1, 1, 2, 2, 3,
                         3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 1, 1, 1,
                         2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10,
                         1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
                         9, 9, 10, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7,
                         7, 8, 8, 9, 9, 10, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3,
                         3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 2, 1, 1,
                         1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
                         8, 8, 9, 9, 10, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4,
                         4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 9, 9, 10, 2, 2, 2, 2,
                         2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
                         8, 9, 9, 10, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5,
                         5, 6, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 3, 3, 3, 3, 3,
                         3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9,
                         9, 10, 10, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 6,
                         6, 7, 7, 7, 8, 8, 9, 9, 10, 10, 10, 4, 4, 4, 4, 4, 4,
                         4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 8, 8, 8, 9, 9, 10,
                         10, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7,
                         8, 8, 8, 9, 9, 10, 10, 10, 5, 5, 5, 5, 5, 5, 5, 5, 6,
                         6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 9, 9, 10, 10, 6, 6, 6,
                         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9,
                         10, 10, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8,
                         8, 9, 9, 9, 10, 10, 10, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                         7, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 7, 7, 7, 7, 7, 7,
                         7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10])

    q_ring_val_m = np.array([[2.5, 4.5],
                             [4.5, 6.5],
                             [6.5, 8.5],
                             [8.5, 10.5],
                             [10.5, 12.5],
                             [12.5, 14.5],
                             [14.5, 16.5],
                             [16.5, 18.5],
                             [18.5, 20.5],
                             [20.5, 22.5]])

    num_pixels_m = np.array([34, 35, 40, 45, 50, 59, 62, 51, 45, 35])

    pixel_list_m = ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                    44, 45, 46, 47, 48, 49, 50, 56, 57, 58, 59, 60, 61,
                    62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
                    75, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
                    93, 94, 95, 96, 97, 98, 99, 100, 106, 107, 108, 109,
                    110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                    121, 122, 123, 124, 125, 131, 132, 133, 134, 135, 136,
                    137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
                    148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158,
                    159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169,
                    170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
                    181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
                    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202,
                    203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213,
                    214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224,
                    225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235,
                    236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246,
                    247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257,
                    258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268,
                    269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279,
                    280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290,
                    291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301,
                    302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312,
                    313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323,
                    324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334,
                    335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345,
                    346, 347, 348, 350, 351, 352, 353, 354, 355, 356, 357,
                    358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368,
                    369, 370, 371, 372, 373, 375, 376, 377, 378, 379, 380,
                    381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391,
                    392, 393, 394, 395, 396, 397, 400, 401, 402, 403, 404,
                    405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415,
                    416, 417, 418, 419, 420, 421, 425, 426, 427, 428, 429,
                    430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440,
                    441, 442, 443, 444, 445, 446, 450, 451, 452, 453, 454,
                    455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465,
                    466, 467, 468, 469, 470, 475, 476, 477, 478, 479, 480,
                    481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491,
                    492, 493, 494]),

    assert_array_almost_equal(q_ring_val_m, q_ring_val)
    assert_array_equal(num_pixels, num_pixels_m)
    assert_array_equal(q_inds, np.ravel(q_inds_m))
    assert_array_equal(pixel_list, pixel_list_m)


def test_roi_rings_step():
    calibrated_center = (4., 4.)
    img_dim = (20, 25)
    first_q = 2.5
    delta_q = 2

    # using a step for the Q rings
    num_qs = 6  # number of q rings
    step_q = 1  # step value between each q ring

    (qstep_inds, qstep_ring_val, numstep_pixels,
     pixelstep_list) = diff_roi.roi_rings_step(img_dim, calibrated_center,
                                               num_qs, first_q, delta_q,
                                               step_q)

    qstep_inds_m = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
                             6, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 1, 1,
                             1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 1, 1, 1, 2, 2,
                             3, 3, 4, 4, 5, 5, 6, 6, 1, 1, 1, 2, 2, 3, 3, 4,
                             4, 5, 5, 6, 6, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                             6, 6, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5,
                             5, 6, 6, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                             6, 6, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 2, 2,
                             2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6,
                             6, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5,
                             6, 6, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 3, 3, 3, 3,
                             3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 3, 3,
                             3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 4, 4,
                             4, 4, 5, 5, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                             4, 5, 5, 5, 5, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                             5, 5, 5, 6, 6, 6, 5, 5, 5, 5, 6, 6, 6, 5, 5, 5,
                             5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 5, 5, 5, 5,
                             5, 5, 5, 5, 5, 6, 6, 6])

    numstep_pixels_m = np.array([34, 35, 45, 57, 62, 47])

    qstep_ring_val_m = np.array([[2.5, 4.5],
                                 [5.5, 7.5],
                                 [8.5, 10.5],
                                 [11.5, 13.5],
                                 [14.5, 16.5],
                                 [17.5, 19.5]])

    pixelstep_list_m = ([0, 1, 2, 3, 4, 5, 6, 9, 10, 12, 13, 15, 16, 18,  19,
                         21, 22, 25, 31, 32, 34, 35, 37, 38, 40, 41, 43, 44,
                         46, 47, 50, 56, 57, 59, 60, 62, 63, 65, 66, 68, 69,
                         71, 72, 75, 81, 82, 84, 85, 87, 88, 90, 91, 93, 94,
                         96, 97, 100, 106, 107, 109, 110, 112, 113, 115, 116,
                         118, 119, 121, 122, 125, 131, 132, 134, 135, 137, 138,
                         140, 141, 143, 144, 146, 147, 150, 151, 152, 153, 154,
                         155, 156, 159, 160, 162, 163, 165, 166, 168, 169, 171,
                         172, 176, 177, 178, 179, 180, 183, 184, 187, 188, 190,
                         191, 193, 194, 196, 197, 207, 208, 209, 211, 212, 214,
                         215, 216, 218, 219, 221, 222, 225, 226, 227, 228, 229,
                         230, 231, 232, 233, 235, 236, 237, 239, 240, 242, 243,
                         245, 246, 247, 250, 251, 252, 253, 254, 255, 256, 259,
                         260, 261, 263, 264, 265, 267, 268, 270, 271, 283, 284,
                         285, 287, 288, 289, 291, 292, 295, 296, 300, 301, 302,
                         303, 304, 305, 306, 307, 308, 309, 311, 312, 313, 315,
                         316, 317, 319, 320, 325, 326, 327, 328, 329, 330, 331,
                         332, 335, 336, 337, 340, 341, 343, 344, 345, 358, 359,
                         360, 361, 364, 365, 368, 369, 375, 376, 377, 378, 379,
                         380, 381, 382, 383, 384, 385, 387, 388, 389, 390, 392,
                         393, 400, 401, 402, 403, 404, 405, 406, 407, 408, 411,
                         412, 413, 416, 417, 418, 434, 435, 436, 437, 440, 441,
                         442, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459,
                         460, 463, 464, 465, 466, 475, 476, 477, 478, 479, 480,
                         481, 482, 483, 487, 488, 489]),

    assert_almost_equal(qstep_ring_val, qstep_ring_val_m)
    assert_array_equal(numstep_pixels, numstep_pixels_m)
    assert_array_equal(qstep_inds, np.ravel(qstep_inds_m))
    assert_array_equal(pixelstep_list, pixelstep_list_m)


def test_roi_rings_diff_steps():
    calibrated_center = (4., 4.)
    img_dim = (25, 15)
    first_q = 2.
    delta_q = 2.

    num_qs = 8  # number of q rings

    (qd_inds, qd_ring_val, numd_pixels,
     pixeld_list) = diff_roi.roi_rings_step(img_dim, calibrated_center, num_qs,
                                            first_q, delta_q, 0.4, 0.2, 0.5,
                                            0.4, 0.0, 0.6, 0.2)

    qd_ring_val_m = np.array([[2., 4.],
                             [4.4, 6.4],
                             [6.6, 8.6],
                             [9.1, 11.1],
                             [11.5, 13.5],
                             [13.5, 15.5],
                             [16.1, 18.1],
                             [18.3, 20.3]])

    numd_pixels_m = np.array([36, 35, 40, 47, 37, 33, 34, 30])

    pixeld_list_m = ([1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15,
                      16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 28,
                      29, 30, 31, 35, 36, 38, 39, 40, 41, 43, 44,
                      45, 46, 50, 51, 53, 54, 55, 56, 58, 59, 60,
                      61, 65, 66, 68, 69, 70, 71, 73, 74, 75, 76,
                      77, 78, 79, 80, 81, 83, 84, 85, 86, 88, 89,
                      91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 103,
                      104, 105, 111, 112, 113, 114, 115, 116, 118,
                      119, 120, 121, 122, 123, 124, 125, 126, 127,
                      128, 129, 130, 132, 133, 134, 135, 136, 137,
                      138, 139, 140, 141, 142, 143, 144, 146, 147,
                      148, 149, 150, 151, 152, 153, 154, 155, 156,
                      157, 158, 161, 162, 163, 164, 165, 166, 167,
                      168, 169, 170, 171, 172, 174, 175, 176, 177,
                      178, 179, 188, 189, 190, 191, 192, 193, 194,
                      195, 196, 197, 198, 199, 200, 201, 202, 203,
                      204, 205, 206, 207, 208, 209, 210, 211, 212,
                      213, 214, 215, 216, 217, 218, 219, 220, 221,
                      222, 223, 224, 225, 226, 227, 228, 229, 230,
                      231, 232, 233, 234, 235, 236, 237, 238, 240,
                      241, 242, 243, 244, 245, 246, 247, 248, 249,
                      250, 251, 252, 254, 255, 256, 257, 258, 259,
                      260, 261, 262, 263, 264, 265, 268, 269, 270,
                      271, 272, 273, 274, 275, 276, 277, 278, 281,
                      282, 283, 284, 294, 295, 296, 297, 298, 299,
                      300, 301, 302, 303, 304, 305, 306, 307, 308,
                      309, 310, 311, 312, 313, 314, 315, 316, 317,
                      318, 319, 320, 321, 322, 323, 324, 325, 326,
                      327, 328, 330, 331, 332, 333, 334, 335, 336,
                      337, 338, 339, 340, 341, 345, 346, 347, 348,
                      349, 350, 351, 352, 353, 354]),

    qd_inds_m = np.array([1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 1, 1,
                          1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 1,
                          1, 2, 2, 3, 3, 4, 4, 1, 1, 1, 1, 2, 2, 3,
                          3, 4, 4, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 1,
                          1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1,
                          1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 2, 2, 2, 2,
                          3, 3, 3, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 3,
                          3, 3, 4, 4, 5, 2, 2, 2, 2, 2, 2, 2, 3, 3,
                          3, 4, 4, 4, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                          4, 4, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4,
                          4, 5, 5, 5, 4, 4, 4, 5, 5, 5, 6, 4, 4, 4,
                          4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 4, 4,
                          4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 5,
                          5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 5,
                          5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 6,
                          6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 6, 6,
                          6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7,
                          7, 7, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                          7, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                          8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                          8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8])

    assert_array_equal(qd_inds, np.ravel(qd_inds_m))
    assert_array_almost_equal(qd_ring_val, qd_ring_val_m)
    assert_array_equal(numd_pixels, numd_pixels_m)
    assert_array_equal(pixeld_list, pixeld_list_m)
