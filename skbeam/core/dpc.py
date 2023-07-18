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
This module is for Differential Phase Contrast (DPC) imaging based on
Fourier shift fitting
"""
from __future__ import absolute_import, division, print_function

import logging
import warnings
from collections import namedtuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def image_reduction(im, roi=None, bad_pixels=None):
    """
    Sum the image data over rows and columns.

    Parameters
    ----------
    im : ndarray
        Input image.

    roi : ndarray, optional
        [r, c, row, col], selects ROI im[r : r + row, c : c + col]. Default is
        None, which uses the whole image.

    bad_pixels : list, optional
        List of (row, column) tuples marking bad pixels.
        [(1, 5), (2, 6)] --> 2 bad pixels --> (1, 5) and (2, 6). Default is
        None.

    Returns
    -------
    xline : ndarray
        The row vector of the sums of each column.

    yline : ndarray
        The column vector of the sums of each row.

    """

    if bad_pixels:
        im = im.copy()
        for row, column in bad_pixels:
            im[row, column] = 0

    if roi:
        r, c, row, col = roi
        im = im[r : (r + row), c : (c + col)]

    xline = np.sum(im, axis=0)
    yline = np.sum(im, axis=1)

    return xline, yline


def _rss_factory(length):
    """
    A factory function for returning a residue function for use in dpc fitting.
    The main reason to do this is to generate a closure over beta so that
    linspace is only called once.

    Parameters
    ----------
    length : int
        The length of the data vector that the returned function can deal with.

    Returns
    -------
    function
        A function with signature f(v, xdata, ydata) which is suitable for use
        as a cost function for use with scipy.optimize.

    """

    beta = 1j * (np.linspace(-(length - 1) // 2, (length - 1) // 2, length))

    def _rss(v, ref_reduction, diff_reduction):
        """
        Internal function used by fit()
        Cost function to be minimized in nonlinear fitting

        Parameters
        ----------
        v : list
            Fit parameters.
            v[0], amplitude of the sample transmission function at one scanning
            point;
            v[1], the phase gradient (along x or y direction) of the sample
            transmission function.

        ref_reduction : ndarray
            Extra argument passed to the objective function. In DPC, it's the
            sum of the reference image data along x or y direction.

        diff_refuction : ndarray
            Extra argument passed to the objective function. In DPC, it's the
            sum of one captured diffraction pattern along x or y direction.

        Returns
        --------
        float
            Residue value.

        """

        diff = diff_reduction - ref_reduction * v[0] * np.exp(v[1] * beta)

        return np.sum((diff * np.conj(diff)).real)

    return _rss


def dpc_fit(rss, ref_reduction, diff_reduction, start_point, solver="Nelder-Mead", tol=1e-6, max_iters=2000):
    """
    Nonlinear fitting for 2 points.

    Parameters
    ----------
    rss : callable
        Objective function to be minimized in DPC fitting.

    ref_reduction : ndarray
        Extra argument passed to the objective function. In DPC, it's the sum
        of the reference image data along x or y direction.

    diff_reduction : ndarray
        Extra argument passed to the objective function. In DPC, it's the sum
        of one captured diffraction pattern along x or y direction.

    start_point : list
        start_point[0], start-searching value for the amplitude of the sample
        transmission function at one scanning point.
        start_point[1], start-searching value for the phase gradient (along x
        or y direction) of the sample transmission function at one scanning
        point.

    solver : str, optional
        Type of solver, one of the following (default 'Nelder-Mead'):
        * 'Nelder-Mead'
        * 'Powell'
        * 'CG'
        * 'BFGS'
        * 'Anneal'
        * 'L-BFGS-B'
        * 'TNC'
        * 'COBYLA'
        * 'SLSQP'

    tol : float, optional
        Termination criteria of nonlinear fitting. Default is 1e-6.

    max_iters : int, optional
        Maximum iterations of nonlinear fitting. Default is 2000.

    Returns
    -------
    tuple
        Fitting result: intensity attenuation and phase gradient.

    """

    return minimize(
        rss,
        start_point,
        args=(ref_reduction, diff_reduction),
        method=solver,
        tol=tol,
        options=dict(maxiter=max_iters),
    ).x


# attributes
dpc_fit.solver = ["Nelder-Mead", "Powell", "CG", "BFGS", "Anneal", "L-BFGS-B", "TNC", "COBYLA", "SLSQP"]


def recon(gx, gy, scan_xstep, scan_ystep, padding=0, weighting=0.5):
    """Reconstruct the final phase image.

    Parameters
    ----------
    gx : ndarray
        Phase gradient along x direction.

    gy : ndarray
        Phase gradient along y direction.

    scan_xstep : float
        Scanning step size in x direction (in micro-meter).

    scan_ystep : float
        Scanning step size in y direction (in micro-meter).

    padding : int, optional

        Pad a N-by-M array to be a
        ``(N*(2*padding+1))``-by-``(M*(2*padding+1))`` array with the
        image in the middle with a (N*padding, M*padding) thick edge
        of zeros. Default is 0.


        padding = 0 --> v (the original image, size = (N, M))
                        0 0 0
        padding = 1 --> 0 v 0 (the padded image, size = (3 * N, 3 * M))
                        0 0 0

    weighting : float, optional
        Weighting parameter for the phase gradient along x and y direction when
        constructing the final phase image.
        Valid in [0, 1]. Default value = 0.5, which means that gx and gy
        equally contribute to the final phase image.

    Returns
    -------
    phase : ndarray
        Final phase image.

    """

    if weighting < 0 or weighting > 1:
        raise ValueError("weighting should be within the range of [0, 1]!")

    pad = 2 * padding + 1
    gx = np.asarray(gx)
    rows, cols = gx.shape
    pad_row = rows * pad
    pad_col = cols * pad

    gx_padding = np.zeros((pad_row, pad_col), dtype="d")
    gy_padding = np.zeros((pad_row, pad_col), dtype="d")

    roi_slice = (slice(padding * rows, (padding + 1) * rows), slice(padding * cols, (padding + 1) * cols))
    gx_padding[roi_slice] = gx
    gy_padding[roi_slice] = gy

    tx = np.fft.fftshift(np.fft.fft2(gx_padding))
    ty = np.fft.fftshift(np.fft.fft2(gy_padding))

    mid_col = pad_col // 2 + 1
    mid_row = pad_row // 2 + 1
    ax = 2 * np.pi * np.arange(1 - mid_col, pad_col - mid_col + 1) / (pad_col * scan_xstep)
    ay = 2 * np.pi * np.arange(1 - mid_row, pad_row - mid_row + 1) / (pad_row * scan_ystep)

    kappax, kappay = np.meshgrid(ax, ay)
    div_v = kappax**2 * (1 - weighting) + kappay**2 * weighting

    with warnings.catch_warnings():
        # It appears that having nans in data arrays is normal mode of
        #   operation for this function. So let's disable warnings.
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        c = -1j * (kappax * tx * (1 - weighting) + kappay * ty * weighting) / div_v
    c = np.fft.ifftshift(np.where(div_v == 0, 0, c))

    phase = np.fft.ifft2(c)[roi_slice].real

    return phase


# holy hacks, Batman!  'index' here is a single element list so
# that I can keep track of how many images have been computed
dpc_internal_state = namedtuple("dpc_internal_state", ["ax", "ay", "gx", "gy", "ref_fx", "ref_fy", "index"])


def dpc_runner(
    ref,
    image_sequence,
    start_point,
    pixel_size,
    focus_to_det,
    scan_rows,
    scan_cols,
    scan_xstep,
    scan_ystep,
    energy,
    padding=0,
    weighting=0.5,
    solver="Nelder-Mead",
    roi=None,
    bad_pixels=None,
    negate=True,
    scale=True,
):
    """Wraps `lazy_dpc`

    See docstring for `lazy_dpc` and `reconstruct_phase_from_partial_info`
    for the input parameters for this function and what it returns
    """
    if len(pixel_size) == 2:
        # make sure the pixels are the same size
        if pixel_size[0] != pixel_size[1]:
            raise ValueError("In DPC, pixels must be square. You provided" "pixel values of {}".format(pixel_size))
    dpc_gen = lazy_dpc(ref, image_sequence, start_point, scan_rows, scan_cols, solver, roi, bad_pixels)
    # exhaust the generator, keeping only the last result
    for dpc_state in dpc_gen:
        pass

    # compute the final results
    phase, amplitude = reconstruct_phase_from_partial_info(
        dpc_state, energy, scan_xstep, scan_ystep, pixel_size[0], focus_to_det, negate, scale, padding, weighting
    )

    return phase, amplitude


def lazy_dpc(
    ref,
    image_sequence,
    start_point,
    scan_rows,
    scan_cols,
    solver="Nelder-Mead",
    roi=None,
    bad_pixels=None,
    dpc_state=None,
):
    """
    Controller function to run the whole Differential Phase Contrast (DPC)
    imaging calculation.

    Parameters
    ----------
    ref : ndarray
        The reference image for a DPC calculation.

    image_sequence : iterable of 2D arrays
        Return diffraction patterns (2D Numpy arrays) when iterated over.

    start_point : list
        start_point[0], start-searching value for the amplitude of the sample
        transmission function at one scanning point.
        start_point[1], start-searching value for the phase gradient (along x
        or y direction) of the sample transmission function at one scanning
        point.

    scan_rows : int
        Number of scanned rows.

    scan_cols : int
        Number of scanned columns.

    solver : str, optional
        Type of solver, one of the following (default 'Nelder-Mead'):
        * 'Nelder-Mead'
        * 'Powell'
        * 'CG'
        * 'BFGS'
        * 'Anneal'
        * 'L-BFGS-B'
        * 'TNC'
        * 'COBYLA'
        * 'SLSQP'

    roi : ndarray, optional
        [r, c, row, col], selects ROI im[r : r + row, c : c + col]. Default is
        None.

    bad_pixels : list, optional
        List of (row, column) tuples marking bad pixels.
        [(1, 5), (2, 6)] --> 2 bad pixels --> (1, 5) and (2, 6). Default is
        None.

    Yields
    ------
    dpc_state : namedtuple
        The internal state that `dpc_runner` requires for each iteration.
        Can be passed to reconstruct_phase_from_partial_info which, along
        with some additional info, will produce the final phase image

    References: text [1]_
    .. [1] Yan, H. et al. Quantitative x-ray phase imaging at the nanoscale by
    multilayer Laue lenses. Sci. Rep. 3, 1307; DOI:10.1038/srep01307 (2013).

    """

    def initialize_state(scan_rows, scan_cols, ref, roi, bad_pixels):
        # Initialize ax, ay, gx, and gy
        ax = np.zeros((scan_rows, scan_cols), dtype="d")
        ay = np.zeros((scan_rows, scan_cols), dtype="d")
        gx = np.zeros((scan_rows, scan_cols), dtype="d")
        gy = np.zeros((scan_rows, scan_cols), dtype="d")
        # Dimension reduction along x and y direction
        refx, refy = image_reduction(ref, roi, bad_pixels)
        ref_fx = np.fft.fftshift(np.fft.ifft(refx))
        ref_fy = np.fft.fftshift(np.fft.ifft(refy))

        return dpc_internal_state(ax, ay, gx, gy, ref_fx, ref_fy, [0])

    if dpc_state is None:
        dpc_state = initialize_state(scan_rows, scan_cols, ref, roi, bad_pixels)

    # 1-D IFFT
    ffx = _rss_factory(len(dpc_state.ref_fx))
    ffy = _rss_factory(len(dpc_state.ref_fy))

    # Same calculation on each diffraction pattern
    for im in image_sequence:
        i, j = np.unravel_index(dpc_state.index[0], (scan_rows, scan_cols))

        # Dimension reduction along x and y direction
        imx, imy = image_reduction(im, roi, bad_pixels)

        # 1-D IFFT
        fx = np.fft.fftshift(np.fft.ifft(imx))
        fy = np.fft.fftshift(np.fft.ifft(imy))

        # Nonlinear fitting
        _ax, _gx = dpc_fit(ffx, dpc_state.ref_fx, fx, start_point, solver)
        _ay, _gy = dpc_fit(ffy, dpc_state.ref_fy, fy, start_point, solver)

        # Store one-point intermediate results
        dpc_state.gx[i, j] = _gx
        dpc_state.gy[i, j] = _gy
        dpc_state.ax[i, j] = _ax
        dpc_state.ay[i, j] = _ay
        dpc_state.index[0] += 1
        yield dpc_state


def reconstruct_phase_from_partial_info(
    dpc_state,
    energy,
    scan_xstep,
    scan_ystep,
    pixel_size=None,
    focus_to_det=None,
    negate=True,
    scale=True,
    padding=0,
    weighting=0.5,
):
    """Using the partial results from dpc_runner, reconstruct the phase image

    Parameters
    ----------
    dpc_state : namedtuple
        The thing yielded from `dpc_runner`
    energy : float
        Energy of the scanning x-ray in keV.
    focus_to_det : float
        Focus to detector distance in um.
    scan_xstep : float
        Scanning step size in x direction (in micro-meter).
    scan_ystep : float
        Scanning step size in y direction (in micro-meter).
    pixel_size : Number, optional
        The size of the detector pixels.  Pixels must be square. If
        `pixel_size and `focus_to_det` are provided, it is assumed that you
        want to scale the image.
    focus_to_det : Number, optional
        The distance from the focal point of the beam to the detector.
        Must be provided as a pair with `pixel_size`.
    negate : bool, optional
        If True (default), negate the phase gradient along x direction before
        reconstructing the final phase image. Default is True.
    scale : bool, optional
        If True, scale gx and gy according to the experiment set up.
        If False, ignore pixel_size, focus_to_det, energy. Default is True.
    padding : int, optional

        Pad a N-by-M array to be a
        ``(N*(2*padding+1))``-by-``(M*(2*padding+1))`` array with the image in
        the middle with a (N*padding, M*padding) thick edge of
        zeros. Default is 0.

        padding = 0 --> v (the original image, size = (N, M))
                        0 0 0
        padding = 1 --> 0 v 0 (the padded image, size = (3 * N, 3 * M))
                        0 0 0
    weighting : float, optional
        Weighting parameter for the phase gradient along x and y direction when
        constructing the final phase image.
        Valid in [0, 1]. Default value = 0.5, which means that gx and gy
        equally contribute to the final phase image.

    Returns
    -------
    phase : ndarray
        The final reconstructed phase image.
    amplitude : ndarray
        Amplitude of the sample transmission function.

    """
    if weighting < 0 or weighting > 1:
        raise ValueError("weighting should be within the range of [0, 1]!")
    gx = None
    gy = dpc_state.gy
    if pixel_size and focus_to_det:
        # Convert to wavelength
        lambda_ = 12.4e-4 / energy
        # pre-compute the scaling factor
        scale = pixel_size / (lambda_ * focus_to_det)
        gx = dpc_state.gx * len(dpc_state.ref_fx) * scale
        gy = dpc_state.gy * len(dpc_state.ref_fy) * scale
    if negate:
        if gx is not None:
            gx *= -1
        else:
            gx = dpc_state.gx * -1
    # Reconstruct the final phase image
    phase = recon(gx, gy, scan_xstep, scan_ystep, padding, weighting)

    return phase, (dpc_state.ax + dpc_state.ay) / 2


# attributes
dpc_runner.solver = ["Nelder-Mead", "Powell", "CG", "BFGS", "Anneal", "L-BFGS-B", "TNC", "COBYLA", "SLSQP"]
