#!/usr/bin/env python
# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
# #
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
This is an example script utilizing dpc.py for Differential Phase Contrast
(DPC) imaging based on Fourier shift fitting.

This script requires a SOFC folder containing the test data in your home
directory. The default path for the results (texts and JPEGs) is also your home
directory. It will automatically download the data to your home directory if
you installed wget and unzip utilities. You can also manually download and
decompress the data at https://www.dropbox.com/s/963c4ymfmbjg5dm/SOFC.zip

Steps
-----
In this file:
    1. Set parameters
    2. Load the reference image
    3. Save intermediate and final results

in skxray.dpc.dpc_runner:
    1. Dimension reduction along x and y direction
    2. 1-D IFFT
    3. Same calculation on each diffraction pattern
        3.1. Read a diffraction pattern
        3.2. Dimension reduction along x and y direction
        3.3. 1-D IFFT
        3.4. Nonlinear fitting
    4. Reconstruct the final phase image
"""

import os
from subprocess import call
import scipy
import numpy as np
import matplotlib.pyplot as plt
from pims import ImageSequence
import zipfile

from skxray.core import dpc
dpc.logger.setLevel(dpc.logging.DEBUG)
handler = dpc.logging.StreamHandler()
handler.setLevel(dpc.logging.DEBUG)
dpc.logger.addHandler(handler)

def load_image(filename):
    """
    Load an image

    Parameters
    ----------
    filename : string
        the location and name of an image

    Return
    ----------
    t : 2-D numpy array
        store the image data

    """

    if os.path.exists(filename):
        t = plt.imread(filename)

    else:
        print('Please download and decompress the test data to your home directory\n\
               Google drive link, https://drive.google.com/file/d/0B3v6W1bQwN_AVjdYdERHUDBsMmM/edit?usp=sharing\n\
               Dropbox link, https://www.dropbox.com/s/963c4ymfmbjg5dm/SOFC.zip')
        raise Exception('File not found: %s' % filename)

    return t


def unzip(source_filename, verbose=True):
    with zipfile.ZipFile(source_filename) as zf:
        num = len(zf.infolist())
        for idx, member in enumerate(zf.infolist()):
            if verbose and idx % (num//100) == 0:
                print("{:3d}% Extracting {}/{}".format(
                      int(idx/num*100), idx+1, len(zf.infolist())))
            zf.extract(member)

def run():
    # download to this folder
    current_folder = os.sep.join(__file__.split(os.sep)[:-1])
    dpc_demo_data_path = os.path.join(current_folder, 'SOFC')

    if not os.path.exists(dpc_demo_data_path):
        sofc_file = os.path.join(current_folder, 'SOFC.zip')
        print('The required test data directory was not found.'
              '\nDownloading the test data to %s' % dpc_demo_data_path)
        # todo make this not print every fraction of a second
        call(('wget https://www.dropbox.com/s/963c4ymfmbjg5dm/SOFC.zip -P %s' %
              current_folder),
             shell=True)
        # unzip it into this directory
        unzip(sofc_file)


    # 1. Set parameters
    start_point = [1, 0]
    first_image = 1
    pixel_size = (55, 55)
    focus_to_det = 1.46e6
    scan_xstep = 0.1
    scan_ystep = 0.1
    scan_rows = 121
    scan_cols = 121
    energy = 19.5
    roi = None
    padding = 0
    weighting = 1.
    bad_pixels = None
    solver = 'Nelder-Mead'
    images = ImageSequence(dpc_demo_data_path + "/*.tif")
    img_size = images[0].shape
    ref_image = np.ones(img_size)
    scale = True
    negate = True

    print('running dpc')
    # 2. Use dpc.dpc_runner
    phase, amplitude = dpc.dpc_runner(
        ref_image, images, start_point, pixel_size, focus_to_det, scan_rows,
        scan_cols, scan_xstep, scan_ystep, energy, padding, weighting, solver,
        roi, bad_pixels, negate, scale)

    # 3. Save intermediate and final results
    scipy.misc.imsave(os.path.join(current_folder, 'phase.jpg'), phase)
    np.savetxt(os.path.join(current_folder, 'phase.txt'), phase)
    scipy.misc.imsave(os.path.join(current_folder, 'amplitude.jpg'), amplitude)
    np.savetxt(os.path.join(current_folder, 'amplitude.txt'), amplitude)


if __name__ == '__main__':
    run()
