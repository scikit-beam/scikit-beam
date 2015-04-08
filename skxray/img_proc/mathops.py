# Module for the BNL image processing project
# Developed at the NSLS-II, Brookhaven National Laboratory
# Developed by Gabriel Iltis, Oct. 2013
"""
This module is designed to facilitate image arithmetic and logical operations
on image data sets.
"""

import numpy as np
import parser


def arithmetic_basic(input_1,
                     input_2,
                     operation):
    """
    This function enables basic arithmetic for image processing and data
    analysis. The function is capable of applying the basic arithmetic
    operations (addition, subtraction, multiplication and division) to two
    data set arrays, two constants, or an array and a constant.

    Parameters
    ----------
    input_1 : {ndarray, int, float}
        Specifies the first input data set, or constant, to be offset or
        manipulated

    input_2 : {ndarray, int, float}
        Specifies the second data set, or constant, to be offset or manipulated

    operation : string
        addition: the addition of EITHER two images or volume data sets,
            OR an image/data set and a value. This function is typically
            used for offset purposes, or basic recombination of several isolated
            materials or phases into a single segmented volume.
        subtraction: enables the subtraction of EITHER one image or volume data
            set from another, OR reduction of all values in an image/data set
            by a set value. This function is typically used for offset
            purposes, or basic isolation of objects or materials/phases in a
            data set.
        multiplication:

        division:


    Returns
    -------
    output : {ndarray, int, float}
        Returns the resulting array or constant to the designated variable

    Example
    -------
    result = mathops.arithmetic_basic(img_1, img_2, 'addition')
    """
    operation_dict = {'addition' : np.add,
                      'subtraction' : np.subtract,
                      'multiplication' : np.multiply,
                      'division' : np.divide
    }
    if operation == 'division':
        if type(input_2) is np.ndarray:
            if 0 in input_2:
                raise ValueError("This division operation will result in "
                                 "division by zero values. Please reevaluate "
                                 "denominator (input_2).")
        else:
            if float(input_2) == 0:
                raise ValueError("This division operation will result in "
                                 "division by a zero value. Please "
                                 "reevaluate the denominator constant"
                                 " (input_2).")

    output = operation_dict[operation](input_1, input_2)
    return output


def arithmetic_custom(expression,
                      A,
                      B,
                      C=None,
                      D=None,
                      E=None,
                      F=None,
                      G=None,
                      H=None):
    """
    This function enables more complex arithmetic to be carried out on 2 or
    more (current limit is 8) arrays or constants. The arithmetic expression
    is defined by the user, as a string, and after assignment of inputs A
    through H the string is parsed into the appropriate python expression
    and executed.  Note that inputs C through H are optional and need only be
    defined when desired or required.


    Parameters
    ----------
    expression : string
        Note that the syntax of the mathematical expression must conform to
        python syntax,
        eg.:
            using * for multiplication instead of x
            using ** for exponents instead of ^

        Arithmetic operators:
            + : addition (adds values on either side of the operator
            - : subtraction (subtracts values on either side of the operator
            * : multiplication (multiplies values on either side of the
                operator
            / : division (divides the left operand (numerator) by the right
                hand operand (denominator))
            % : modulus (divides the left operand (numerator) by the right
                hand operand (denominator) and returns the remainder)
            ** : exponent (left operand (base) is raised to the power of the
                 right operand (exponent))
            // : floor division (divides the left operand (numerator) by the
                 right hand operand (denominator), but returns the quotient
                 with any digits after the decimal point removed,
                 e.g. 9.0/2.0 = 4.0)

        Logical operations are also included and available so long as the:
            > : greater than
            < : less than
            == : exactly equals
            != : not equal
            >= : greater than or equal
            <= : less than or equal

        Additional operators:
            = : assignment operator (assigns values from right side to those
                on the left side)
            += : Adds the right operand to the left operand and sets the
                 total equal to the left operand,
                 e.g.:
                    b+=a is equivalent to b=a+b
            -= : Subtracts the right operand from the left operand and sets
                 the total equal to the left operand,
                 e.g.:
                    b -= a is equivalent to b = b - a
            *= : multiplies the right operand to the left operand and sets the
                 total equal to the left operand,
                 e.g.:
                    b *= a is equivalent to b = b * a
            /= : divides the right operand into the left operand and sets the
                 total equal to the left operand,
                 e.g.:
                    b /= a is equivalent to b = b / a
            %= : divides the right operand into the left operand and sets the
                 remainder equal to the left operand,
                 e.g.:
                    b %= a is equivalent to b =b % a
            **= : raises the left operand to the power of the right operand
                  and sets the total equal to the left operand,
                  e.g.:
                    b **= a is equivalent to b = b ** a
            //= : divides the right operand into the left operand and
                  then removes any values after the decimal point. The total
                  is then set equal to the left operand,
                 e.g.:
                    b //= a is equivalent to b = b // a

        In the event that bitwise operations are required the operators &,
        |, ^, ~ may also be used, though I'm struggling to come up with a
        scenario where this will be used.

        Order of operations and parenthesis are taken into account when
        evaluating the expression.

    A : {ndarray, int, float}
        Data set or constant to be offset or manipulated

    B : {ndarray, int, float}
        Data set or constant to be offset or manipulated

    C : {ndarray, int, float}, optional
        Data set or constant to be offset or manipulated

    D : {ndarray, int, float}, optional
        Data set or constant to be offset or manipulated

    E : {ndarray, int, float}, optional
        Data set or constant to be offset or manipulated

    F : {ndarray, int, float}, optional
        Data set or constant to be offset or manipulated

    G : {ndarray, int, float}, optional
        Data set or constant to be offset or manipulated

    H : {ndarray, int, float}, optional
        Data set or constant to be offset or manipulated


    Returns
    -------
    output : {ndarray, int, float}
        Returns the resulting array or value to the designated variable

    Example
    -------
    result = mathops.arithmetic_custom('(A+C)/(B+D)', img_1, img_2, 2, 4)
    """
    output = eval(parser.expr(expression).compile())
    return output


def logic_basic(operation,
                src_data1,
                src_data2=None):
    """
    This function enables the computation of the basic logical operations
    oft used in image processing of two image or volume  data sets. This
    function can be used for data comparison, material isolation,
    noise removal, or mask application/generation.

    Parameters
    ----------
    operation : str
        options include:
            'and' -- 2 inputs
            'or' -- 2 inputs
            'not' -- 1 input
            'xor' -- 2 inputs
            'nand' -- 2 inputs
            'subtract' -- 2 inputs

    src_data1 : {ndarray, int, float, list, tuple}
        Specifies the first reference

    src_data2 : {ndarray, int, float, list, tuple}
        Specifies the second reference

    Returns
    -------
    output : {ndarray, bool}
        Returns the result of the logical operation, which can be an array,
        or a simple boolean result.

    Example
    -------
    result = mathops.logic_basic('and', img_1, img_2)
    """
    logic_dict = {'and' : np.logical_and,
                  'or' : np.logical_or,
                  'not' : np.logical_not,
                  'xor' : np.logical_xor,
                  'nand' : logical_NAND,
                  'nor' : logical_NOR,
                  'subtract' : logical_SUB
                  }
    output = logic_dict[operation](src_data1,
                                   src_data2)
    return output


def logical_NAND(src_data1,
                 src_data2):
    """
    This function enables the computation of the LOGICAL_NAND of two image or 
    volume data sets. This function enables easy isolation of all data points 
    NOT INCLUDED IN BOTH SOURCE DATA SETS. This function can be used for data 
    comparison, material isolation, noise removal, or mask 
    application/generation.

    Parameters
    ----------
    src_data1 : ndarray
        Specifies the first reference data

    src_data2 : ndarray
        Specifies the second reference data

    Returns
    -------
    output : ndarray
        Returns the resulting array to the designated variable

    Example
    -------
    result = mathops.logical_NAND('img_1', 'img_2')
    """
    output = np.logical_not(np.logical_and(src_data1, 
                                           src_data2))
    return output


def logical_NOR(src_data1,
                 src_data2):
    """
    This function enables the computation of the LOGICAL_NOR of two image or
    volume data sets. This function enables easy isolation of all data points
    NOT INCLUDED IN EITHER OF THE SOURCE DATA SETS. This function can be used
    for data comparison, material isolation, noise removal, or mask
    application/generation.

    Parameters
    ----------
    src_data1 : ndarray
        Specifies the first reference data

    src_data2 : ndarray
        Specifies the second reference data

    Returns
    -------
    output : ndarray
        Returns the resulting array to the designated variable

    Example
    -------
    result = mathops.logical_NOR('img_1', 'img_2')
    """
    output = np.logical_not(np.logical_or(src_data1,
                                          src_data2))
    return output


def logical_SUB(src_data1,
                src_data2):
    """
    This function enables LOGICAL SUBTRACTION of one binary image or volume data 
    set from another. This function can be used to remove phase information, 
    interface boundaries, or noise, present in two data sets, without having to 
    worry about mislabeling of pixels which would result from arithmetic 
    subtraction. This function will evaluate as true for all "true" voxels 
    present ONLY in Source Dataset 1. This function can be used for data 
    cleanup, or boundary/interface analysis.

    Parameters
    ----------
    src_data1 : ndarray
        Specifies the first reference data

    src_data2 : ndarray
        Specifies the second reference data

    Returns
    -------
    output : ndarray
        Returns the resulting array to the designated variable

    Example
    -------
    result = mathops.logical_SUB('img_1', 'img_2')
    """
    temp = np.logical_not(np.logical_and(src_data1, 
                                         src_data2))
    output = np.logical_and(src_data1, 
                            temp)
    return output
