
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
from numpy.testing import (assert_equal, assert_array_almost_equal)
from skxray.fitting.base.parameter_data import default_param
from skxray.fitting.xrf_model import ModelSpectrum


def test_xrf_model():
    MS = ModelSpectrum(default_param)
    MS.model_spectrum()
    assert_equal(len(MS.mod.components), 8)
    #return MS.mod

