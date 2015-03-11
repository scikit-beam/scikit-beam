
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import six
import numpy as np
from numpy.testing import (assert_equal, assert_array_almost_equal)
from skxray.fitting.base.parameter_data import get_para, default_param
from skxray.fitting.xrf_model import ModelSpectrum, ParamController


def test_xrf_model():
    MS = ModelSpectrum()
    MS.model_spectrum()
    assert_equal(len(MS.mod.components), 24)


def test_parameter_controller():
    param = get_para()
    PC = ParamController(param)
    PC.create_full_param()
    PC.update_element_prop(['Fe', 'Ce_L'],
                           pos='fixed', width='lohi',
                           area='none', ratio='fixed')
    PC.set_bound_type('linear')

    for k, v in six.iteritems(PC.new_parameter):

        if 'Fe' in k:
            if 'ratio' in k or 'center' in k:
                assert_equal(str(v['bound_type']), 'fixed')
            elif 'area' in k:
                assert_equal(str(v['bound_type']), 'none')
            elif 'sigma' in k:
                assert_equal(str(v['bound_type']), 'lohi')
