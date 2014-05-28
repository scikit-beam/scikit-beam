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
from six import string_types
from collections import namedtuple, MutableMapping, Counter

md_value = namedtuple("md_value", ['value', 'units'])


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


keys_core = {"voxel_size": "description of voxel_size",
             "detector_center_x": "x-coordinate of the center of the image plate in pixels",
             "detector_center_y": "y-coordinate of the center of the image plate in pixels",
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
    Exception
        Possible causes:
            is_reference contains no true values
            Raised when the first image in the array is not a reference image.

    """

    counts = Counter(is_reference)
    if counts[1] == 0:
        raise Exception("There are no reference images in is_reference")

    if is_reference[0] is not True:
        raise Exception("The first image is not a reference image")

    img_corr = [None] * (len(img_arr) - counts[1])
    img_corr_pos = 0
    for img, ref in zip(img_arr, is_reference):
        if ref:
            cur_ref = img
        else:
            img_corr[img_corr_pos] = img - cur_ref
            img_corr_pos += 1

    return img_corr
