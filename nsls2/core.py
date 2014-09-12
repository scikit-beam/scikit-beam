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
This module is for the 'core' data types.
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
from six.moves import zip
from six import string_types

import time
import sys
from itertools import tee
from collections import namedtuple, MutableMapping

import numpy as np

import logging
logger = logging.getLogger(__name__)

try:
    import src.ctrans as ctrans
except ImportError:
    try:
        import ctrans
    except ImportError:
        ctrans = None


md_value = namedtuple("md_value", ['value', 'units'])


_defaults = {
    "bins": 100,
    'nx': 100,
    'ny': 100,
    'nz': 100
}


class MD_dict(MutableMapping):
    """
    A class to make dealing with the meta-data scheme for DataExchange easier

    Examples
    --------
    Getting and setting data by path is possible

    >>> tt = MD_dict()
    >>> tt['name'] = 'test'
    >>> tt['nested.a'] = 2
    >>> tt['nested.b'] = (5, 'm')
    >>> tt['nested.a'].value
    2
    >>> tt['nested.a'].units is None
    True
    >>> tt['name'].value
    'test'
    >>> tt['nested.b'].units
    'm'
    """
    def __init__(self, md_dict=None):
        # TODO properly walk the input on upgrade dicts -> MD_dict
        if md_dict is None:
            md_dict = dict()

        self._dict = md_dict
        self._split = '.'

    def __repr__(self):
        return self._dict.__repr__()

    # overload __setitem__ so dotted paths work
    def __setitem__(self, key, val):

        key_split = key.split(self._split)
        tmp = self._dict
        for k in key_split[:-1]:
            try:
                tmp = tmp[k]._dict
            except:
                tmp[k] = type(self)()
                tmp = tmp[k]._dict
            if isinstance(tmp, md_value):
                # TODO make message better
                raise KeyError("trying to use a leaf node as a branch")

        # if passed in an md_value, set it and return
        if isinstance(val, md_value):
            tmp[key_split[-1]] = val
            return
        # catch the case of a bare string
        elif isinstance(val, string_types):
            # a value with out units
            tmp[key_split[-1]] = md_value(val, 'text')
            return
        # not something easy, try to guess what to do instead
        try:
            # if the second element is a string or None, cast to named tuple
            if isinstance(val[1], string_types) or val[1] is None:
                print('here')
                tmp[key_split[-1]] = md_value(*val)
            # else, assume whole thing is the value with no units
            else:
                tmp[key_split[-1]] = md_value(val, None)
        # catch any type errors from trying to index into non-indexable things
        # or from trying to use iterables longer than 2
        except TypeError:
            tmp[key_split[-1]] = md_value(val, None)

    def __getitem__(self, key):
        key_split = key.split(self._split)
        tmp = self._dict
        for k in key_split[:-1]:
            try:
                tmp = tmp[k]._dict
            except:
                tmp[k] = type(self)()
                tmp = tmp[k]._dict

            if isinstance(tmp, md_value):
                # TODO make message better
                raise KeyError("trying to use a leaf node as a branch")

        return tmp.get(key_split[-1], None)

    def __delitem__(self, key):
        # pass one delete the entry
        # TODO make robust to non-keys
        key_split = key.split(self._split)
        tmp = self._dict
        for k in key_split[:-1]:
            # make sure we are grabbing the internal dict
            tmp = tmp[k]._dict
        del tmp[key_split[-1]]
        # TODO pass 2 remove empty branches

    def __len__(self):
        return len(list(iter(self)))

    def __iter__(self):
        return _iter_helper([], self._split, self._dict)


class verbosedict(dict):
    def __getitem__(self, key):
        try:
            v = dict.__getitem__(self, key)
        except KeyError:
            if len(self) < 25:
                new_msg = ("You tried to access the key '{key}' "
                           "which does not exist.  The "
                           "extant keys are: {valid_keys}").format(
                               key=key, valid_keys=list(self))
            else:
                new_msg = ("You tried to access the key '{key}' "
                           "which does not exist.  There "
                           "are {num} extant keys, which is too many to "
                           "show you").format(
                               key=key, num=len(self))
            six.reraise(KeyError, new_msg, sys.exc_info()[2])
        return v


def _iter_helper(path_list, split, md_dict):
    """
    Recursively walk the tree and return the names of the leaves
    """
    for k, v in six.iteritems(md_dict):
        if isinstance(v, md_value):
            yield split.join(path_list + [k])
        else:
            for inner_v in _iter_helper(path_list + [k], split, v._dict):
                yield inner_v


keys_core = {
    "pixel_size": {
        "description": ("2 element tuple defining the (x y) dimensions of the "
                        "pixel"),
        "type": tuple,
        "units": "um",
    },
    "voxel_size": {
        "description": ("3 element tuple defining the (x y z) dimensions of "
                        "the voxel"),
        "type": tuple,
        "units": "um",
    },
    "calibrated_center": {
        "description": ("2 element tuple defining the (x y) center of the "
                        "detector in pixels"),
        "type": tuple,
        "units": "pixel",
    },
    "detector_size": {
        "description": ("2 element tuple defining no. of pixels(size) in the "
                        "detector X and Y direction"),
        "type": tuple,
        "units": "pixel",
    },
    "detector_tilt_angles": {
        "description": "Detector tilt angle",
        "type": tuple,
        "units": " degrees",
    },
    "dist_sample": {
        "description": "distance from the sample to the detector (mm)",
        "type": float,
        "units": "mm",
    },
    "wavelength": {
        "description": "wavelength of incident radiation (Angstroms)",
        "type": float,
        "units": "angstrom",
    },
    "ub_mat": {
        "description": "UB matrix(orientation matrix) 3x3 array",
        "type": "ndarray",
    },
}


def img_subtraction_pre(img_arr, is_reference):
    """
    Function to subtract a series of measured images from
    background/dark current/reference images.  The nearest reference
    image in the reverse temporal direction is subtracted from each
    measured image.

    Parameters
    ----------
    img_arr : numpy.ndarray
        Array of 2-D images

    is_reference : 1-D boolean array
        true  : image is reference image
        false : image is measured image

    Returns
    -------
    img_corr : numpy.ndarray
        len(img_corr) == len(img_arr) - len(is_reference_img == true)
        img_corr is the array of measured images minus the reference
        images.

    Raises
    ------
    ValueError
        Possible causes:
            is_reference contains no true values
            Raised when the first image in the array is not a reference image.

    """
    # an array of 1, 0, 1,.. should work too
    if not is_reference[0]:
        # use ValueError because the user passed in invalid data
        raise ValueError("The first image is not a reference image")
    # grab the first image
    ref_imge = img_arr[0]
    # just sum the bool array to get count
    ref_count = np.sum(is_reference)
    # make an array of zeros of the correct type
    corrected_image = np.zeros(
        (len(img_arr) - ref_count,) + img_arr.shape[1:],
        dtype=img_arr.dtype)
    # local loop counter
    count = 0
    # zip together (lazy like this is really izip), images and flags
    for img, ref in zip(img_arr[1:], is_reference[1:]):
        # if this is a ref image, save it and move on
        if ref:
            ref_imge = img
            continue
        # else, do the subtraction
        corrected_image[count] = img - ref_imge
        # and increment the counter
        count += 1

    # return the output
    return corrected_image


def detector2D_to_1D(img, detector_center, **kwargs):
    """
    Convert the 2D image to a list of x y I coordinates where
    x == x_img - detector_center[0] and
    y == y_img - detector_center[1]

    Parameters
    ----------
    img: ndarray
        2D detector image
    detector_center: 2 element array
        see keys_core["detector_center"]["description"]
    **kwargs: dict
        Bucket for extra parameters in an unpacked dictionary

    Returns
    -------
    X : numpy.ndarray
        1 x N
        x-coordinate of pixel
    Y : numpy.ndarray
        1 x N
        y-coordinate of pixel
    I : numpy.ndarray
        1 x N
        intensity of pixel
    """

    # Caswell's incredible terse rewrite
    X, Y = np.meshgrid(np.arange(img.shape[0]) - detector_center[0],
                       np.arange(img.shape[1]) - detector_center[1])

    # return the x, y and z coordinates (as a tuple? or is this a list?)
    return X.ravel(), Y.ravel(), img.ravel()


def bin_1D(x, y, nx=None, min_x=None, max_x=None):
    """
    Bin the values in y based on their x-coordinates

    Parameters
    ----------
    x : array
        position
    y : array
        intensity
    nx : integer, optional
        number of bins to use
    min_x : float, optional
        Left edge of first bin
    max_x : float, optional
        Right edge of last bin

    Returns
    -------
    edges : array
        edges of bins, length nx + 1

    val : array
        sum of values in each bin, length nx

    count : array
        The number of counts in each bin, length nx
    """

    # handle default values
    if min_x is None:
        min_x = np.min(x)
    if max_x is None:
        max_x = np.max(x)
    if nx is None:
        nx = _defaults["bins"]

    # use a weighted histogram to get the bin sum
    bins = np.linspace(start=min_x, stop=max_x, num=nx+1, endpoint=True)
    val, _ = np.histogram(a=x, bins=bins, weights=y)
    # use an un-weighted histogram to get the counts
    count, _ = np.histogram(a=x, bins=bins)
    # return the three arrays
    return bins, val, count


def radial_integration(img, detector_center, sample_to_detector_distance,
                       pixel_size, wavelength):
    """
    docstring!
    """
    pass


def wedge_integration(src_data, center, theta_start,
                      delta_theta, r_inner, delta_r):
    """
    Implementation of caking.

    Parameters
    ----------
    scr_data : ndarray
        The source-data to be integrated

    center : ndarray
        The center of the ring in pixels

    theta_start : float
        The angle of the start of the wedge from the
        image y-axis in degrees

    delta_theta : float
        The angular width of the wedge in degrees.  Positive
        angles go clockwise, negative go counter-clockwise.

    r_inner : float
        The inner radius in pixel units, Must be non-negative

    delta_r : float
        The length of the wedge in the radial direction
        in pixel units. Must be non-negative

    Returns
    -------
    float
        The integrated intensity under the wedge
    """
    pass


def bin_edges(range_min=None, range_max=None, nbins=None, step=None):
    """
    Generate bin edges.  The last value is the returned array is
    the right edge of the last bin, the rest of the values are the
    left edges of each bin.

    If `range_max` is specified all bin edges will be less than or
    equal to it's value.

    If `range_min` is specified all bin edges will be greater than
    or equal to it's value

    If `nbins` is specified then there will be than number of bins and
    the returned array will have length `nbins + 1` (as the right most
    edge is included)

    If `step` is specified then bin width is approximately `step` (It is
    not exact due to the nature of floats). The arrays generated by
    `np.cumsum(np.ones(nbins) * step)` and `np.arange(nbins) * step` are
    not identical.  This function uses the second method in all cases
    where `step` is specified.

    .. warning :: If the set :code:`(range_min, range_max, step)` is
        given there is no guarantee that :code:`range_max - range_min`
        is an integer multiple of :code:`step`.  In this case the left
        most bin edge is :code:`range_min` and the right most bin edge
        is less than :code:`range_max` and the distance between the
        right most edge and :code:`range_max` is not greater than
        :code:`step` (this is the same behavior as the built-in
        :code:`range()`).  It is not recommended to specify bins in this
        manner.

    Parameters
    ----------
    range_min : float, optional
        The minimum value that may be included as a bin edge

    range_max : float, optional
        The maximum value that may be included as a bin edge

    nbins : int, optional
        The number of bins, if specified the length of the returned
        value will be nbins + 1

    step : float, optional
        The step between the bins

    Returns
    -------
    np.array
        An array of floats for the bin edges.  The last value is the
        right edge of the last bin.
    """
    num_valid_args = sum((range_min is not None, range_max is not None,
                          step is not None, nbins is not None))
    if num_valid_args != 3:
        raise ValueError("Exactly three of the arguments must be non-None "
                         "not {}.".format(num_valid_args))

    if range_min is not None and range_max is not None:
        if range_max <= range_min:
            raise ValueError("The minimum must be less than the maximum")

    if nbins is not None:
        if nbins <= 0:
            raise ValueError("The number of bins must be positive")

    # The easy case
    if step is None:
        return np.linspace(range_min, range_max, nbins + 1, endpoint=True)

    # in this case, the user gave use min, max, and step
    if nbins is None:
        if step > (range_max - range_min):
            raise ValueError("The step can not be greater than the difference "
                             "between min and max")
        nbins = int((range_max - range_min)//step)
        ret = range_min + np.arange(nbins + 1) * step
        # if the last value is greater than the max (should never happen)
        if ret[-1] > range_max:
            return ret[:-1]
        if range_max - ret[-1] > 1e-10 * step:
            logger.debug("Inconsistent "
                         "(range_min, range_max, step) "
                         "and step does not evenly divide "
                         "(range_min - range_max). "
                         "The bins has been truncated.\n"
                         "min: %f max: %f step: %f gap: %f",
                         range_min, range_max,
                         step, range_max - ret[-1])
        return ret

    # in this case we got range_min, nbins, step
    if range_max is None:
        return range_min + np.arange(nbins + 1) * step

    # in this case we got range_max, nbins, step
    if range_min is None:
        return range_max - np.arange(nbins + 1)[::-1] * step


def grid3d(q, img_stack,
           nx=None, ny=None, nz=None,
           xmin=None, xmax=None, ymin=None,
           ymax=None, zmin=None, zmax=None):
    """Grid irregularly spaced data points onto a regular grid via histogramming

    This function will process the set of reciprocal space values (q), the
    image stack (img_stack) and grid the image data based on the bounds
    provided, using defaults if none are provided.

    Parameters
    ----------
    q : ndarray
        (Qx, Qy, Qz) - HKL values - Nx3 array
    img_stack : ndarray
        Intensity array of the images
        dimensions are: [num_img][num_rows][num_cols]
    nx : int, optional
        Number of voxels along x
    ny : int, optional
        Number of voxels along y
    nz : int, optional
        Number of voxels along z
    xmin : float, optional
        Minimum value along x. Defaults to smallest x value in q
    ymin : float, optional
        Minimum value along y. Defaults to smallest y value in q
    zmin : float, optional
        Minimum value along z. Defaults to smallest z value in q
    xmax : float, optional
        Maximum value along x. Defaults to largest x value in q
    ymax : float, optional
        Maximum value along y. Defaults to largest y value in q
    zmax : float, optional
        Maximum value along z. Defaults to largest z value in q

    Returns
    -------
    mean : ndarray
        intensity grid.  The values in this grid are the
        mean of the values that fill with in the grid.
    occupancy : ndarray
        The number of data points that fell in the grid.
    std_err : ndarray
        This is the standard error of the value in the
        grid box.
    oob : int
        Out Of Bounds. Number of data points that are outside of
        the gridded region.
    bounds : list
        tuple of (min, max, step) for x, y, z in order: [x_bounds,
        y_bounds, z_bounds]

    """

    q = np.atleast_2d(q)
    q.shape
    if q.ndim != 2:
        raise ValueError("q.ndim must be a 2-D array of shape Nx3 array. "
                         "You provided an array with {0} dimensions."
                         "".format(q.ndim))
    if q.shape[1] != 3:
        raise ValueError("The shape of q must be an Nx3 array, not {0}X{1}"
                         " which you provided.".format(*q.shape))

    # set defaults for qmin, qmax, dq
    qmin = np.min(q, axis=0)
    qmax = np.max(q, axis=0)
    dqn = [_defaults['nx'], _defaults['ny'], _defaults['nz']]

    # pad the upper edge by just enough to ensure that all of the
    # points are in-bounds with the binning rules: lo <= val < hi
    qmax += np.spacing(qmax)

    # check for non-default input
    for target, input_vals in ((dqn, (nx, ny, nz)),
                               (qmin, (xmin, ymin, zmin)),
                               (qmax, (xmax, ymax, zmax))):
        for j, in_val in enumerate(input_vals):
            if in_val is not None:
                target[j] = in_val

    # format bounds
    bounds = np.array([qmin, qmax, dqn]).T

    # creating (Qx, Qy, Qz, I) Nx4 array - HKL values and Intensity
    # getting the intensity value for each pixel
    q = np.insert(q, 3, np.ravel(img_stack), axis=1)

    #            3D grid of the data set
    # starting time for gridding
    t1 = time.time()

    # call the c library
    mean, occupancy, std_err, oob = ctrans.grid3d(q, qmin, qmax, dqn, norm=1)

    # ending time for the gridding
    t2 = time.time()
    logger.info("Done processed in {0} seconds".format(t2-t1))

    # No. of values zero in the grid
    empt_nb = (occupancy == 0).sum()

    # log some information about the grid at the debug level
    if oob:
        logger.debug("There are %.2e points outside the grid {0}".format(oob))
    logger.debug("There are %2e bins in the grid {0}".format(mean.size))
    if empt_nb:
        logger.debug("There are %.2e values zero in the grid {0}"
                     "".format(empt_nb))

    return mean, occupancy, std_err, oob, bounds


def bin_edges_to_centers(input_edges):
    """
    Helper function for turning a array of bin edges into
    an array of bin centers

    Parameters
    ----------
    input_edges : array-like
        N + 1 values which are the left edges of N bins
        and the right edge of the last bin

    Returns
    -------
    ndarray
        A length N array giving the centers of the bins
    """
    input_edges = np.asarray(input_edges)
    return (input_edges[:-1] + input_edges[1:]) * 0.5
