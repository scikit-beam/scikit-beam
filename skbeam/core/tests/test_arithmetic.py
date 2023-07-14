# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Sept. 2014
"""
This module contains test functions for the file-IO functions
for reading and writing data sets using the netCDF file format.

The files read and written using this function are assumed to
conform to the format specified for x-ray computed microtomorgraphy
data collected at Argonne National Laboratory, Sector 13, GSECars.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
from numpy.testing import assert_equal

from skbeam.core import arithmetic


def _helper_prealloc_passthrough(op, x1, x2, scratch_space):
    ret = op(x1, x2, scratch_space)
    assert_equal(ret, scratch_space)


@pytest.mark.parametrize("op", [arithmetic.logical_nand, arithmetic.logical_sub, arithmetic.logical_nor])
def test_prealloc_passthrough(op):
    """Smoketest the pre-allocation bit of numpy"""
    x1 = np.arange(10)
    x2 = np.arange(10)
    scratch_space = np.zeros(x1.shape)
    _helper_prealloc_passthrough(op, x1, x2, scratch_space)


def test_logical_nor():
    test_array_1 = np.zeros((30, 30, 30), dtype=int)
    test_array_1[0:15, 0:15, 0:15] = 1
    test_array_2 = np.zeros((30, 30, 30), dtype=int)
    test_array_2[15:29, 15:29, 15:29] = 87
    test_array_3 = np.ones((30, 30, 30), dtype=float)
    test_array_3[10:20, 10:20, 10:20] = 87.4
    test_array_4 = np.zeros((30, 30), dtype=int)
    test_array_4[24:29, 24:29] = 254

    test_1D_array_1 = np.zeros(100, dtype=int)
    test_1D_array_1[0:30] = 50
    test_1D_array_2 = np.zeros(50, dtype=int)
    test_1D_array_2[20:49] = 10

    assert_equal(arithmetic.logical_nor(test_array_1, test_array_1), np.logical_not(test_array_1))
    assert_equal(
        arithmetic.logical_nor(test_array_1, test_array_2).sum(),
        (np.ones((30, 30, 30), dtype=int).sum() - (np.logical_or(test_array_1, test_array_2).sum())),
    )


def test_logical_nand():
    # TEST DATA
    test_array_1 = np.zeros((90, 90, 90), dtype=int)
    test_array_1[10:19, 10:19, 10:19] = 1
    test_array_2 = np.zeros((90, 90, 90), dtype=int)
    test_array_2[20:29, 20:29, 20:29] = 2
    test_array_3 = np.zeros((90, 90, 90), dtype=int)
    test_array_3[30:39, 30:39, 30:39] = 3
    test_array_4 = np.zeros((90, 90, 90), dtype=int)
    test_array_4[40:49, 40:49, 40:49] = 4
    test_array_5 = np.zeros((90, 90, 90), dtype=int)
    test_array_5[50:59, 50:59, 50:59] = 5
    test_array_6 = np.zeros((90, 90, 90), dtype=int)
    test_array_6[60:69, 60:69, 60:69] = 6
    test_array_7 = np.zeros((90, 90, 90), dtype=int)
    test_array_7[70:79, 70:79, 70:79] = 7
    test_array_8 = np.zeros((90, 90, 90), dtype=int)
    test_array_8[80:89, 80:89, 80:89] = 8

    assert_equal(arithmetic.logical_nand(test_array_1, test_array_1), np.logical_not(test_array_1))
    test_result = arithmetic.logical_nand(test_array_1, test_array_2)
    assert_equal(test_result[20:39, 20:39, 20:39], True)


def test_logical_sub():
    # TEST DATA
    test_array_1 = np.zeros((90, 90, 90), dtype=int)
    test_array_1[0:39, 0:39, 0:39] = 1
    test_array_2 = np.zeros((90, 90, 90), dtype=int)
    test_array_2[20:79, 20:79, 20:79] = 2
    test_array_3 = np.zeros((90, 90, 90), dtype=int)
    test_array_3[40:89, 40:89, 40:89] = 3

    assert_equal(arithmetic.logical_sub(test_array_1, test_array_1), False)
    test_result = arithmetic.logical_sub(test_array_1, test_array_2)
    assert_equal(test_result[20:39, 20:39, 20:39], False)
    assert_equal(test_result.sum(), (test_array_1.sum() - np.logical_and(test_array_1, test_array_2).sum()))

    test_result = arithmetic.logical_sub(test_array_1, test_array_3)
    assert_equal(test_result, test_array_1)
