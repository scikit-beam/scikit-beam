# Copyright (c) Brookhaven National Lab 2O14
# All rights reserved
# BSD License
# See LICENSE for full text
"""
This module is for the 'core' data types.
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from six.moves import zip
from six import string_types
from collections import namedtuple, MutableMapping
import numpy as np

md_value = namedtuple("md_value", ['value', 'units'])


_defaults = {
    "bins" : 100,
}


class XR_data(object):
    """
    A class for wrapping up and carrying around data + unrelated
    meta data.
    """
    def __init__(self, data, md=None, mutable=True):
        """
        Parameters
        ----------
        data : object
            The 'data' object to be carried around

        md : dict
            The meta-data object, needs to support [] access


        """
        self._data = data
        if md is None:
            md = dict()
        self._md = md
        self.mutable = mutable

    @property
    def data(self):
        """
        Access to the data object we are carrying around
        """
        return self._data

    @data.setter
    def data(self, new_data):
        if not self.mutable:
            raise RuntimeError("Can't set data on immutable instance")
        self._data = new_data

    def __getitem__(self, key):
        """
        Over-ride the [] infrastructure to access the meta-data

        Parameters
        ----------
        key : hashable object
            The meta-data key to retrive
        """
        return self._md[key]

    def __setitem__(self, key, val):
        """
        Over-ride the [] infrastructure to access the meta-data

        Parameters
        ----------
        key : hashable object
            The meta-data key to set

        val : object
            The new meta-data value to set
        """
        if not self.mutable:
            raise RuntimeError("Can't set meta-data on immutable instance")
        self._md[key] = val

    def meta_data_keys(self):
        """
        Get a list of the meta-data keys that this object knows about
        """
        return list(six.iterkeys(self._md))


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
        "description" : ("2 element tuple defining the (x y) dimensions of the "
                         "pixel"),
        "type" : tuple,
        "units" : "um",
        },
    "voxel_size": {
        "description" : ("3 element tuple defining the (x y z) dimensions of the "
                         "voxel"),
        "type" : tuple,
        "units" : "um",
        },
     "detector_center": {
        "description" : ("2 element tuple defining the (x y) center of the "
                         "detector in pixels"),
        "type" : tuple,
        "units" : "pixel",
        },
     "dist_sample": {
        "description" : "distance from the sample to the detector (mm)",
        "type" : float,
        "units" : "mm",
        },
     "wavelength": {
        "description" : "wavelength of incident radiation (Angstroms)",
        "type" : float,
        "units" : "angstrom",
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
    x : 1D array-like
        position
    y : 1D array-like
        intensity
    nx : integer
        number of bins to use
    min_x : float
        Left edge of first bin
    max_x : float
        Right edge of last bin

    Returns
    -------
    edges : 1D array
        edges of bins, length nx + 1

    val : 1D array
        sum of values in each bin, length nx

    count : ID
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
