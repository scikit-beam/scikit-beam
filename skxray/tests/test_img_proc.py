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

import numpy as np
import six
from skxray.img_proc import mathops
from numpy.testing import assert_equal, assert_raises


#Test Data
test_array_1 = np.zeros((30,30,30), dtype=int)
test_array_1[0:15, 0:15, 0:15] = 1
test_array_2 = np.zeros((50, 70, 50), dtype=int)
test_array_2[25:50, 25:50, 25:50] = 87
test_array_3 = np.zeros((10,10,10), dtype=float)
test_array_4 = np.zeros((100,100,100), dtype=float)
test_array_5 = np.zeros((100,100), dtype=int)
test_array_5[25:75, 25:75] = 254

test_1D_array_1 = np.zeros((100), dtype=int)
test_1D_array_2 = np.zeros((10), dtype=int)
test_1D_array_3 = np.zeros((100), dtype=float)

test_constant_int = 5
test_constant_flt = 2.0
test_constant_bool = True

def test_arithmetic_basic():
    """
    Test function for the image processing function: arithmetic_basic

    """
    test_array_1 = np.zeros((30,30,30), dtype=int)
    test_array_1[0:15, 0:15, 0:15] = 1
    test_array_2 = np.zeros((30, 30, 30), dtype=int)
    test_array_2[15:29, 15:29, 15:29] = 87
    test_array_3 = np.ones((30,30,30), dtype=float)
    test_array_3[10:20, 10:20, 10:20] = 87.4
    test_array_4 = np.zeros((30,30), dtype=int)
    test_array_4[24:29, 24:29] = 254

    test_1D_array_1 = np.zeros((100), dtype=int)
    test_1D_array_1[0:30]=50
    test_1D_array_2 = np.zeros((50), dtype=int)
    test_1D_array_2[20:49]=10
    test_1D_array_3 = np.ones((100), dtype=float)

    test_constant_int = 5
    test_constant_flt = 2.0

    #Int array and int constant
    add_check = test_array_1 + test_constant_int
    sub_check = np.subtract(test_array_1, test_constant_int)
    mult_check = np.multiply(test_array_1, test_constant_int)
    div_check = np.divide(test_array_1, test_constant_int)

    assert_equal(mathops.arithmetic_basic(test_array_1,
                                          test_constant_int,
                                          'addition'),
                 add_check)
    assert_equal(mathops.arithmetic_basic(test_array_1,
                                          test_constant_int,
                                          'subtraction'),
                 sub_check)
    assert_equal(mathops.arithmetic_basic(test_array_1,
                                          test_constant_int,
                                          'multiplication'),
                 mult_check)
    assert_equal(mathops.arithmetic_basic(test_array_1,
                                          test_constant_int,
                                          'division'),
                 div_check)
    assert_raises(ValueError,
                  mathops.arithmetic_basic,
                  test_array_1,
                  test_array_1,
                  'division')
    assert_raises(ValueError,
                  mathops.arithmetic_basic,
                  test_array_1,
                  0,
                  'division')

    #Int array and int array
    add_check = test_array_1 + test_array_2
    sub_check = np.subtract(test_array_1, test_array_2)
    mult_check = np.multiply(test_array_1, test_array_2)
    div_check = np.divide(test_array_1, test_array_2)

    assert_equal(mathops.arithmetic_basic(test_array_1,
                                          test_array_2,
                                          'addition'),
                 add_check)
    assert_equal(mathops.arithmetic_basic(test_array_1,
                                          test_array_2,
                                          'subtraction'),
                 sub_check)
    assert_equal(mathops.arithmetic_basic(test_array_1,
                                          test_array_2,
                                          'multiplication'),
                 mult_check)
    assert_raises(ValueError,
                  mathops.arithmetic_basic,
                  test_array_2,
                  test_array_1,
                  'division')

    #Float array and float constant
    add_check = test_array_3 + test_constant_flt
    sub_check = np.subtract(test_array_3, test_constant_flt)
    mult_check = np.multiply(test_array_3, test_constant_flt)
    div_check = np.divide(test_array_3, test_constant_flt)

    assert_equal(mathops.arithmetic_basic(test_array_3,
                                          test_constant_flt,
                                          'addition'),
                 add_check)
    assert_equal(mathops.arithmetic_basic(test_array_3,
                                          test_constant_flt,
                                          'subtraction'),
                 sub_check)
    assert_equal(mathops.arithmetic_basic(test_array_3,
                                          test_constant_flt,
                                          'multiplication'),
                 mult_check)
    assert_equal(mathops.arithmetic_basic(test_array_3,
                                          test_constant_flt,
                                          'division'),
                 div_check)
    #Float array and float array
    add_check = test_array_3 + test_array_3
    sub_check = np.subtract(test_array_3, test_array_3)
    mult_check = np.multiply(test_array_3, test_array_3)
    div_check = np.divide(test_array_3, test_array_3)

    assert_equal(mathops.arithmetic_basic(test_array_3,
                                          test_array_3,
                                          'addition'),
                 add_check)
    assert_equal(mathops.arithmetic_basic(test_array_3,
                                          test_array_3,
                                          'subtraction'),
                 sub_check)
    assert_equal(mathops.arithmetic_basic(test_array_3,
                                          test_array_3,
                                          'multiplication'),
                 mult_check)
    assert_equal(mathops.arithmetic_basic(test_array_3,
                                          test_array_3,
                                          'division'),
                 div_check)
    #Mixed dtypes: Int array and float array
    assert_equal(mathops.arithmetic_basic(test_array_1,
                                          test_array_1.astype(float),
                                          'addition').dtype,
                  float)
    #Float array and int constant
    assert_equal(mathops.arithmetic_basic(test_array_3,
                                          test_constant_int,
                                          'addition').dtype,
                  float)
    #Int array and float constant
    assert_equal(mathops.arithmetic_basic(test_array_1,
                                          test_constant_flt,
                                          'addition').dtype,
                  float)
    #Mismatched array sizes
    assert_raises(ValueError,
                  mathops.arithmetic_basic,
                  test_array_1,
                  test_array_3,
                  'addition')


def test_arithmetic_custom():
    """
    Test function for mathops.arithmetic_custom, a function that allows the
    inclusion of up to 8 inputs (arrays or constants) and application of a
    custom expression, to simplify image arithmetic including 2 or more
    objects or parameters.
    """
    #TEST DATA
    test_array_1 = np.zeros((90,90,90), dtype=int)
    test_array_1[10:19, 10:19, 10:19] = 1
    test_array_2 = np.zeros((90,90,90), dtype=int)
    test_array_2[20:29, 20:29, 20:29] = 2
    test_array_3 = np.zeros((90,90,90), dtype=int)
    test_array_3[30:39, 30:39, 30:39] = 3
    test_array_4 = np.zeros((90,90,90), dtype=int)
    test_array_4[40:49, 40:49, 40:49] = 4
    test_array_5 = np.zeros((90,90,90), dtype=int)
    test_array_5[50:59, 50:59, 50:59] = 5
    test_array_6 = np.zeros((90,90,90), dtype=int)
    test_array_6[60:69, 60:69, 60:69] = 6
    test_array_7 = np.zeros((90,90,90), dtype=int)
    test_array_7[70:79, 70:79, 70:79] = 7
    test_array_8 = np.zeros((90,90,90), dtype=int)
    test_array_8[80:89, 80:89, 80:89] = 8

    #Array manipulation
    #-int only
    result = (test_array_1 + test_array_2 + test_array_3 + test_array_4 +
              test_array_5 + test_array_6 + test_array_7 + test_array_8)
    assert_equal(mathops.arithmetic_custom('A+B+C+D+E+F+G+H',
                                           test_array_1,
                                           test_array_2,
                                           test_array_3,
                                           test_array_4,
                                           test_array_5,
                                           test_array_6,
                                           test_array_7,
                                           test_array_8),
                 result)
    #-float only
    result = ((test_array_1.astype(float) + 3.5) +
              (test_array_3.astype(float) / 2.0) -
              test_array_4.astype(float)
    )
    assert_equal(mathops.arithmetic_custom('(A+B)+(C/D)-E',
                                           test_array_1.astype(float),
                                           3.5,
                                           test_array_3.astype(float),
                                           2.0,
                                           test_array_4.astype(float)),
                 result)
    #-mixed int and float
    result = ((test_array_1 + 3.5) +
              (test_array_3.astype(float) / 2) -
              test_array_4
    )
    assert_equal(mathops.arithmetic_custom('(A+B)+(C/D)-E',
                                           test_array_1,
                                           3.5,
                                           test_array_3.astype(float),
                                           2,
                                           test_array_4.astype(float)),
                 result)
    assert_equal(mathops.arithmetic_custom('(A+B)+(C/D)-E',
                                           test_array_1,
                                           3.5,
                                           test_array_3.astype(float),
                                           2,
                                           test_array_4.astype(float)).dtype,
                 float)



def test_logic():
    """
    Test function for mathops.logic_basic, a function that allows for
    logical operations to be performed on one or two arrays or constants
    depending on the type of operation.
    For example:
    logical not only takes one object, and returns the inverse, while the
    other operations provide a comparison of two objects).
    """
    #TEST DATA
    test_array_1 = np.zeros((90,90,90), dtype=int)
    test_array_1[0:39, 0:39, 0:39] = 1
    test_array_2 = np.zeros((90,90,90), dtype=int)
    test_array_2[20:79, 20:79, 20:79] = 2
    test_array_3 = np.zeros((90,90,90), dtype=int)
    test_array_3[40:89, 40:89, 40:89] = 3

    #and
    assert_equal(mathops.logic_basic('and', test_array_1, test_array_1),
                 test_array_1)

    test_result = mathops.logic_basic('and', test_array_1, test_array_2)
    assert_equal(test_result[20:39, 20:39, 20:39], True)
    assert_equal(test_result.sum(), ((39-20)**3))

    test_result = mathops.logic_basic('and', test_array_1, test_array_3)
    assert_equal(test_result, False)

    #or
    assert_equal(mathops.logic_basic('or', test_array_1, test_array_1),
                 test_array_1)

    assert_equal(mathops.logic_basic('or',
                                     test_array_1,
                                     test_array_2).sum(),
                 (test_array_1.sum() +
                  test_array_2.sum() /
                  2 -
                  np.logical_and(test_array_1,
                                 test_array_2).sum()
                 )
    )
    test_result = mathops.logic_basic('or', test_array_1, test_array_3)
    assert_equal(test_result.sum(),
                 (test_array_1.sum() +
                  test_array_3.sum() /
                  test_array_3.max()
                 )
    )

    #not
    assert_equal(mathops.logic_basic('not', test_array_1).sum(),
                 (90**3-test_array_1.sum()))
    assert_equal(mathops.logic_basic('not', test_array_3).sum(),
                 (90**3-(test_array_3.sum()/test_array_3.max())))

    #xor
    assert_equal(mathops.logic_basic('xor', test_array_1, test_array_1),
                 np.zeros((90,90,90), dtype=int))
    assert_equal(mathops.logic_basic('xor',
                                     test_array_1,
                                     test_array_2).sum(),
                 ((test_array_1.sum() +
                  test_array_2.sum() / 2) -
                  (2 * np.logical_and(test_array_1,
                                      test_array_2).sum()
                  )
                 )
    )

    #nand
    assert_equal(mathops.logic_basic('nand', test_array_1, test_array_1),
                 np.logical_not(test_array_1))
    test_result = mathops.logic_basic('nand', test_array_1, test_array_2)
    assert_equal(test_result[20:39, 20:39, 20:39], False)

    #nor
    assert_equal(mathops.logic_basic('nor', test_array_1, test_array_1),
                 np.logical_not(test_array_1))
    assert_equal(mathops.logic_basic('nor',
                                     test_array_1,
                                     test_array_2).sum(),
                 (np.ones((90,90,90), dtype=int).sum() -
                  (np.logical_or(test_array_1,
                                 test_array_2).sum()
                  )
                  )
                 )

    #subtract
    assert_equal(mathops.logic_basic('subtract', test_array_1, test_array_1),
                 False)
    test_result = mathops.logic_basic('subtract', test_array_1, test_array_2)
    assert_equal(test_result[20:39, 20:39, 20:39], False)
    assert_equal(test_result.sum(),
                 (test_array_1.sum() -
                  np.logical_and(test_array_1,
                                 test_array_2).sum()
                 )
    )
    test_result = mathops.logic_basic('subtract', test_array_1, test_array_3)
    assert_equal(test_result, test_array_1)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
