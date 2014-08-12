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
import numpy as np

import six
import logging
logger = logging.getLogger(__name__)


def read_binary(filename, nx, ny, nz, dsize, headersize, **kwargs):
    """
    docstring, woo!

    Parameters
    ----------
    filename: String
              The name of the file to open
    nx: integer
        The number of data elements in the x-direction
    ny: integer
        The number of data elements in the y-direction
    nz: integer
        The number of data elements in the z-direction
    dsize: numpy data type
           The size of each element in the numpy array
    headersize: integer
                The size of the file header in bytes
    extras: dict
            unnecessary keys that were passed to this function through
            dictionary unpacking

    Returns
    -------
    (data, header)
    data: ndarray
            data.shape = (x, y, z) if z > 1
            data.shape = (x, y) if z == 1
            data.shape = (x,) if y == 1 && z == 1
    header: String
            header = file.read(headersize)
    """

    # open the file
    opened_file = open(filename, "rb")

    # read the file header
    header = opened_file.read(headersize)

    # read the entire file in as 1D list
    data = np.fromfile(opened_file, dsize, -1)

    # reshape the array to 3D
    if nz is not 1:
        data.resize(nx, ny, nz)
    # unless the 3rd dimension is 1, in which case reshape the array to 2D
    elif ny is not 1:
        data.resize(nx, ny)
    # unless the 2nd dimension is also 1, in which case leave the array as 1D

    # return the array and the header
    return data, header
