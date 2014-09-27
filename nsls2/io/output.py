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


def save_chi(tth, intensity,  filename, err=None, dir_path=None):
    """
    Save output diffraction intensities into .chi file format

    Parameters
    ----------
    tth : ndarray
        2(theta) values or (Q values) 1XN array

    intensity : ndarray
        intensity values 1XN array

    filename : str
        filename of the .tif file

    err : ndarray, optional
        uncertainty

    dir_path : str, optional
        new directory path to save the output data files

    Returns
    -------
    output : str
             saved file of diffraction intensities in .chi file format
    """
    if len(tth) != len(intensity):
        raise ValueError("Number of intensities and the number of Q or"
                         " two theta values are different ")

    file_base = os.path.splitext(os.path.split(filename)[1])[0]

    if len(dir_path) > 0 and not os.path.exists(dir_path):
        raise ValueError('The given path does not exist.')
    if not os.path.isabs(dir_path):
        print 'Creating image: ' + os.path.join(os.getcwd(), dir_path, filename)
        file_path = dir_path + file_base + '.chi'
    else:
        print 'Creating image: ' + os.path.join(dir_path, filename)
        file_path = file_base + '.chi'


    f = open(file_path, 'wb')
    f.write(filename)
    f.write("This file contains integrated powder x-ray diffraction "
            "intensities.")
    f.write("First two columns represents Q(reciprocal space) or 2(theta)"
            " values and second column represents intensities and if there"
            " is third column it represents  intensities error")
    f.write("############################################################\n")

    if (err==None):
        np.savetxt(f, (tth, intensity), newline='\n')
    else:
        np.savetxt(f, (tth, intensity, err), newline='\n')
    f.close()
    return


def save_dat(tth, intensity, filename, err=None, dir_path=None):
    '''
    Save output diffraction intensities into .dat file format

    Parameters
    ----------
    tth : ndarray
        2(theta) values or (Q values)

    intensity : ndarray
        intensity values

    filename : str
        filename(could be full path) of the .tif file

    err : ndarray, optional
        uncertainty

    dir_path : str, optional
        name of the new directory to save the output data files

    Returns
    -------
    output : file
        Saved file of diffraction intensities in .dat file format
    '''
    if len(tth) != len(intensity):
        raise ValueError("Number of intensities and the number of Q or"
                         " two theta values are different ")

    file_base = os.path.splitext(os.path.split(filename)[1])[0]
    file_path = file_base + '.dat'

    f = open(file_path, 'wb')
    f.write(filename)
    f.write(filename)
    f.write("This file contains integrated powder x-ray diffraction "
            "intensities.")
    f.write("First two columns represents Q(reciprocal space) or 2(theta)"
            " values and second column represents intensities and if there"
            " is third column it represents  intensities error")
    f.write("#############################################################\n")

    if (err==None):
        np.savetxt(f, (tth, intensity), newline='\n')
    else:
        if len(tth)==len(err):
            np.savetxt(f, (tth, intensity, err), newline='\n')
        else:
            raise ValueError("")
    f.close()

    return


def save_xye(tth, intensity, err, filename, dir_name=None):
    """
    Save diffraction intensities into .xye file format

    Parameters
    ----------
    tth : ndarray
        2(theta) values or (Q values)

    intensity : ndarray
        intensity values

    err : ndarray
        uncertainty

    filename : str
        filename(could be full path) of the .tif file

     dir_path : str, optional
        new directory path to save the output data files

    Returns
    -------
    output : file
             Saved file of diffraction intensities in .xye file format
    """
    if len(tth) != len(intensity):
        raise ValueError("Number of intensities and the number of Q or"
                         " two theta values are different ")

    file_base = os.path.splitext(os.path.split(filename)[1])[0]
    file_path = file_base + '.xye'

    f = open(file_path, 'wb')
    f.write(filename)
    f.write("This file contains integrated powder x-ray diffraction"
            " intensities.")
    f.write("First two columns represents Q(reciprocal space) "
            "or 2(th eta) values, second column represents"
            " intensities and third column represents intensities error")
    f.write("#########################################################\n")

    np.savetxt(f, (tth, intensity, err), newline='\n')
    f.close()

    return


def save_gsas(tth, intensity, filename, err=None, dir_name=None):
    """
    Save diffraction intensities into .gsas file format

    Parameters
    ----------
    tth : ndarray
        2(theta) values or (Q values)

    intensity : ndarray
        intensity values

    path : str
        directory to save the chi files

    filename : str
        filename(could be full path) of the .tif file

    err : ndarray, None
        uncertainty

     dir_path : str, optional
        new directory path to save the output data files


    Returns
    -------
    output : str
        Saved file of diffraction intensities in .gsas file format

    """

 def saveGSAS(self, xrd, filename):
        '''
        save diffraction intensity in gsas format

        :param xrd: 2d array with shape (2,len of intensity) or (3, len of intensity), [tthorq, intensity, (unceratinty)]
        :param filename: str, base file name
        '''
        if len(tth) != len(intensity):
            raise ValueError("Number of intensities and the number of Q or"
                         " two theta values are different ")

        filepath = self.getFilePathWithoutExt(filename) + '.gsas'
        f = open(filepath, 'wb')
        f.write(self.config.getHeader(mode='short'))
        f.write('#### start data\n')
        if xrd.shape[0] == 3:
            s = writeGSASStr(os.path.splitext(path)[0], self.gsasoutput, xrd[0], xrd[1], xrd[2])
        elif xrd.shape[0] == 2:
            s = writeGSASStr(os.path.splitext(path)[0], self.gsasoutput, xrd[0], xrd[1])
        f.write(s)
        f.close()
        return filepath

def writeGSASStr(name, mode, tth, iobs, esd=None):
    """
    Return string of integrated intensities in GSAS format.
    :param mode: string, gsas file type, could be 'std', 'esd', 'fxye' (gsas format)
    :param tth: ndarray, two theta angle
    :param iobs: ndarray, Xrd intensity
    :param esd: ndarray, optional error value of intensity

    :return:  string, a string to be saved to file
    """
    maxintensity = 999999
    logscale = numpy.floor(numpy.log10(maxintensity / numpy.max(iobs)))
    logscale = min(logscale, 0)
    scale = 10 ** int(logscale)
    lines = []
    ltitle = 'Angular Profile'
    ltitle += ': %s' % name
    ltitle += ' scale=%g' % scale
    if len(ltitle) > 80:    ltitle = ltitle[:80]
    lines.append("%-80s" % ltitle)
    ibank = 1
    nchan = len(iobs)
    # two-theta0 and dtwo-theta in centidegrees
    tth0_cdg = tth[0] * 100
    dtth_cdg = (tth[-1] - tth[0]) / (len(tth) - 1) * 100
    if esd == None: mode = 'std'
    if mode == 'std':
        nrec = int(numpy.ceil(nchan / 10.0))
        lbank = "BANK %5i %8i %8i CONST %9.5f %9.5f %9.5f %9.5f STD" % \
                (ibank, nchan, nrec, tth0_cdg, dtth_cdg, 0, 0)
        lines.append("%-80s" % lbank)
        lrecs = [ "%2i%6.0f" % (1, ii * scale) for ii in iobs ]
        for i in range(0, len(lrecs), 10):
            lines.append("".join(lrecs[i:i + 10]))
    if mode == 'esd':
        nrec = int(numpy.ceil(nchan / 5.0))
        lbank = "BANK %5i %8i %8i CONST %9.5f %9.5f %9.5f %9.5f ESD" % \
                (ibank, nchan, nrec, tth0_cdg, dtth_cdg, 0, 0)
        lines.append("%-80s" % lbank)
        lrecs = [ "%8.0f%8.0f" % (ii, ee * scale) for ii, ee in zip(iobs, esd) ]
        for i in range(0, len(lrecs), 5):
            lines.append("".join(lrecs[i:i + 5]))
    if mode == 'fxye':
        nrec = nchan
        lbank = "BANK %5i %8i %8i CONST %9.5f %9.5f %9.5f %9.5f FXYE" % \
                (ibank, nchan, nrec, tth0_cdg, dtth_cdg, 0, 0)
        lines.append("%-80s" % lbank)
        lrecs = [ "%22.10f%22.10f%24.10f" % (xx * scale, yy * scale, ee * scale) for xx, yy, ee in zip(tth, iobs, esd) ]
        for i in range(len(lrecs)):
            lines.append("%-80s" % lrecs[i])
    lines[-1] = "%-80s" % lines[-1]
    rv = "\r\n".join(lines) + "\r\n"
    return rv

