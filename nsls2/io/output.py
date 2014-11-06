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
    (Output into different file formats, .chi, .dat, .xye, .gsas)

"""

import numpy as np
import scipy.io
import os


def save_output(tth, intensity,  filename, q_or_2theta, ext=None, err=None,
                dir_path=None):
    """
    Save output diffraction intensities into .chi, .dat or .xye file formats.
    If the extension(ext) of the output file is not selected it will be
    saved as a .chi file

    Parameters
    ----------
    tth : ndarray
        twotheta values in degrees or Q values in Angstrom
        shape 1XN array

    intensity : ndarray
        intensity values 1XN array

    filename : str
        filename(could be full path) to the image data
        eg: /Data/BeamLines/tiff_files/BCNS/BCNCS_260K-00504.tif
        or /Data/BeamLines/tiff_files/BCNS/BCNCS_260K-00504

    q_or_2theta : str
        twotheta or Q values (enter "Q" or "2theta")
        twotheta space = "2theta"
        Q space = "Q"

    ext : str, optional
        save output diffraction intensities into .chi, .dat  or
        .xye file formats.(enter ".chi", ".dat", "xye")

    err : ndarray, optional
         error value of intensity

    dir_path : str, optional
        new directory path to save the output data files
        eg: /Volumes/Data/experiments/data/

    Returns
    -------
    Saved file of diffraction intensities in .chi, .dat or .xye
     file formats

    """
    if len(tth) != len(intensity):
        raise ValueError("Number of intensities and the number of Q or"
                         " two theta values are different ")

    file_base = os.path.splitext(os.path.split(filename)[1])[0]

    # the extension of output file is not selected it will be
    # saved as a .chi file

    if ext == None:
        ext = '.chi'

    if ext == '.xye':
        if err == None:
            raise ValueError("Provide the Error value of intensity"
                             " ( for .xye file format err != None )")

    if (dir_path) == None:
        file_path = file_base + ext
    elif os.path.isabs(dir_path):
        file_path = dir_path + file_base + ext
    else:
        raise ValueError('The given path does not exist.')

    with open(file_path, 'wb') as f:
        f.write(filename)

        f.write("\n This file contains integrated powder x-ray diffraction "
            "intensities.\n\n")

        f.write("Number of data points in the file {0} \n".format(len(tth)))

        if q_or_2theta == "Q":
            f.write("First column represents Q values (Angstroms) and second column \n"
                    "represents intensities and if there is a third column it \n"
                    "represents the error value of intensities\n")

        elif q_or_2theta == "2theta":
            f.write("First column represents two theta values (degrees) and \n"
                    "second column represents intensities and if there is \n"
                    "a third column it represents the error value of "
                    " intensities\n")

        else:
           raise ValueError("It is expected to provide whether the data is"
                          " Q values(enter Q) or two theta values"
                          " (enter 2theta)")
        f.write("#####################################################\n\n")

        if (err == None):
            np.savetxt(f, np.c_[tth, intensity], newline='\n')
        else:
            np.savetxt(f, np.c_[tth, intensity, err], newline='\n')

    return


def save_gsas(tth, intensity, filename, mode=None, err=None, dir_path=None):
    """
    Save diffraction intensities into .gsas file format

    Parameters
    ----------
    tth : ndarray
        2(theta) values

    intensity : ndarray
        intensity values

    filename : str
        filename(could be full path) of the .tif file
        eg: /Data/BeamLines/tiff_files/BCNS/BCNCS_260K-00504.tif

    mode : str, optional
        gsas file type, could be 'std', 'esd', 'fxye' (gsas format)

    err : ndarray, optional
        error value of intensity

     dir_path : str, optional
        new directory path to save the output data files
        eg: /Data/experiments/data/

    Returns
    -------
    Saved file of diffraction intensities in .gsas file format

    """
    if len(tth) != len(intensity):
        raise ValueError("Number of intensities and the number of"
                         " two theta values are different ")

    file_base = os.path.splitext(os.path.split(filename)[1])[0]

    if (dir_path) == None:
        file_path = file_base +'.gsas'
    elif os.path.isabs(dir_path):
        file_path = dir_path + file_base +'.gsas'
    else:
        raise ValueError('The given path does not exist.')

    max_intensity = 999999
    log_scale = np.floor(np.log10(max_intensity / np.max(intensity)))
    log_scale = min(log_scale, 0)
    scale = 10 ** int(log_scale)
    lines = []

    title = 'Angular Profile'
    title += ': %s' % filename
    title += ' scale=%g' % scale
    if len(title) > 80:
        title = title[:80]
    lines.append("%-80s" % title)
    i_bank = 1
    n_chan = len(intensity)

    # two-theta0 and dtwo-theta in centidegrees
    tth0_cdg = tth[0] * 100
    dtth_cdg = (tth[-1] - tth[0]) / (len(tth) - 1) * 100

    if err == None:
        mode = 'std'

    if mode == 'std':
        n_rec = int(np.ceil(n_chan / 10.0))
        l_bank = "BANK %5i %8i %8i CONST %9.5f %9.5f %9.5f %9.5f STD" % \
                (i_bank, n_chan, n_rec, tth0_cdg, dtth_cdg, 0, 0)
        lines.append("%-80s" % l_bank)
        lrecs = ["%2i%6.0f" % (1, ii * scale) for ii in intensity]
        for i in range(0, len(lrecs), 10):
            lines.append("".join(lrecs[i:i + 10]))
    elif mode == 'esd':
        n_rec = int(np.ceil(n_chan / 5.0))
        l_bank = "BANK %5i %8i %8i CONST %9.5f %9.5f %9.5f %9.5f ESD" % \
                (i_bank, n_chan, n_rec, tth0_cdg, dtth_cdg, 0, 0)
        lines.append("%-80s" % l_bank)
        l_recs = ["%8.0f%8.0f" % (ii, ee * scale) for ii,
                                                      ee in zip(intensity,
                                                                err)]
        for i in range(0, len(l_recs), 5):
                lines.append("".join(l_recs[i:i + 5]))
    elif mode == 'fxye':
        n_rec = n_chan
        l_bank = "BANK %5i %8i %8i CONST %9.5f %9.5f %9.5f %9.5f FXYE" % \
                (i_bank, n_chan, n_rec, tth0_cdg, dtth_cdg, 0, 0)
        lines.append("%-80s" % l_bank)
        l_recs = ["%22.10f%22.10f%24.10f" % (xx * scale,
                                              yy * scale,
                                              ee * scale) for xx,
                                                              yy,
                                                              ee in zip(tth,
                                                                        intensity,
                                                                        err)]
        for i in range(len(l_recs)):
            lines.append("%-80s" % l_recs[i])
    else:
        raise ValueError("  Define the GSAS file type   ")

    lines[-1] = "%-80s" % lines[-1]
    rv = "\r\n".join(lines) + "\r\n"

    with open(file_path, 'wb') as f:
        f.write(rv)

    return
