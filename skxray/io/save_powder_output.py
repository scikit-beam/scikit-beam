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
    This module is for saving integrated powder x-ray diffraction
    intensities into  different file formats.
    (Output into different file formats, .chi, .dat and .xye)

"""
from __future__ import (absolute_import, division,
                        unicode_literals, print_function)
import numpy as np
import scipy.io
import os
import sys
import logging
logger = logging.getLogger(__name__)


def save_output(tth, intensity, output_name, q_or_2theta, ext='.chi',
                err=None, dir_path=None):
    """
    Save output diffraction intensities into .chi, .dat or .xye file formats.
    If the extension(ext) of the output file is not selected it will be
    saved as a .chi file

    Parameters
    ----------
    tth : ndarray
        twotheta values (degrees) or Q values (Angstroms)
        shape (N, ) array

    intensity : ndarray
        intensity values (N, ) array

    output_name : str
        name for the saved output diffraction intensities

    q_or_2theta : {'Q', '2theta'}
        twotheta (degrees) or Q (Angstroms) values

    ext : {'.chi', '.dat', '.xye'}, optional
        save output diffraction intensities into .chi, .dat  or
        .xye file formats. (If the extension of output file is not
        selected it will be saved as a .chi file)

    err : ndarray, optional
         error value of intensity shape (N, ) array

    dir_path : str, optional
        new directory path to save the output data files
        eg: /Volumes/Data/experiments/data/
    """

    if q_or_2theta not in set(['Q', '2theta']):
        raise ValueError("It is expected to provide whether the data is"
                         " Q values(enter Q) or two theta values"
                         " (enter 2theta)")

    if q_or_2theta == "Q":
        des = ("\n First column represents Q values (Angstroms) and second"
               " column represents intensities and if there is a third"
               " column it represents the error value of intensities")
    else:
        des = ("\n First column represents two theta values (degrees) and"
               "  second column represents intensities and if there is"
               " a third column it represents the error value of intensities")

    _validate_input(tth, intensity, err, ext)

    file_path = _create_file_path(dir_path, output_name, ext)

    with open(file_path, 'wb') as f:
        f.write((output_name).encode('utf-8'))

        f.write(("\nThis file contains integrated powder x-ray diffraction "
            "intensities.\n\n").encode('utf-8'))

        f.write(("Number of data points in the"
                 " file : {0} \n".format(len(tth))).encode('utf-8'))

        f.write((des).encode('utf-8'))

        f.write(("\n #############################################"
                 "###########################################\n\n").encode('utf-8'))

        if (err is None):
            np.savetxt(f, np.c_[tth, intensity])
        else:
            np.savetxt(f, np.c_[tth, intensity, err])


def _validate_input(tth, intensity, err, ext):
    """
    This function validate all the inputs and create a output file
    path to save diffraction intensities

    Parameters
    ----------
    tth : ndarray
        twotheta values (degrees) or Q space values (Angstroms)

    intensity : ndarray
        intensity values

    err : ndarray, optional
        error value of intensity

    ext : {'.chi', '.dat', '.xye'}
        save output diffraction intensities into .chi,
        .dat or .xye file formats.
    """

    if len(tth) != len(intensity):
        raise ValueError("Number of intensities and the number of Q or"
                         " two theta values are different ")

    if ext == '.xye' and err is None:
        raise ValueError("Provide the Error value of intensity"
                         " (for .xye file format err != None)")


def _create_file_path(dir_path, output_name, ext):
    """
    This function create a output file path to save
    diffraction intensities.

    Parameters
    ----------
    dir_path : str
        new directory path to save the output data files
        eg: /Data/experiments/data/

    output_name : str
        name for the saved output diffraction intensities

    ext : {'.chi', '.dat', '.xye'}
        save output diffraction intensities into .chi,
        .dat or .xye file formats.

    Returns:
    -------
    file_path : str
        path to save the diffraction intensities
    """

    if (dir_path) is None:
        file_path = output_name + ext
    elif os.path.exists(dir_path):
        file_path = os.path.join(dir_path, output_name) + ext
    else:
        raise ValueError('The given path does not exist.')

    if os.path.isfile(file_path):
        logger.info("Output file of diffraction intensities"
                    " already exists")
        os.remove(file_path)

    return file_path
