
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import six
import numpy as np
from numpy.testing import (assert_equal, assert_array_almost_equal)
from nose.tools import assert_true, raises
from skxray.fitting.base.parameter_data import get_para, default_param
from skxray.fitting.xrf_model import (ElementModel, ModelSpectrum,
                                      ParamController, _set_parameter_hint,
                                      pre_fit_linear,
                                      get_linear_model, set_range, PreFitAnalysis)


def synthetic_spectrum():
    param = get_para()
    x = np.arange(2000)

    elist, matv = get_linear_model(x, param, default_area=1e5)
    return np.sum(matv, 1) + 100  # avoid zero values


# def test_xrf_model():
#     MS = ModelSpectrum()
#     MS.model_spectrum()
#     assert_equal(len(MS.mod.components), 24)


def test_parameter_controller():
    param = get_para()
    PC = ParamController(param)
    PC.create_full_param()
    set_opt = dict(pos='hi', width='lohi', area='hi', ratio='lo')
    PC.update_element_prop(['Fe', 'Ce_L'], **set_opt)
    PC.set_bound_type('linear')

    # check boundary value
    for k, v in six.iteritems(PC.new_parameter):
        if 'Fe' in k:
            if 'ratio' in k:
                assert_equal(str(v['bound_type']), set_opt['ratio'])
            if 'center' in k:
                assert_equal(str(v['bound_type']), set_opt['pos'])
            elif 'area' in k:
                assert_equal(str(v['bound_type']), set_opt['area'])
            elif 'sigma' in k:
                assert_equal(str(v['bound_type']), set_opt['width'])


def test_fit():
    x0 = np.arange(2000)
    y0 = synthetic_spectrum()

    x, y = set_range(x0, y0, 100, 1300)
    MS = ModelSpectrum()
    MS.model_spectrum()

    result = MS.model_fit(x, y, w=1/np.sqrt(y), maxfev=200)
    for k, v in six.iteritems(result.values):
        if 'area' in k:
            # error smaller than 1%
            assert_true((v-1e5)/1e5 < 1e-2)


def test_pre_fit():
    y0 = synthetic_spectrum()

    # the following items should appear
    item_list = ['Ar_K', 'Fe_K', 'compton', 'elastic']

    param = get_para()

    # with weight pre fit
    x, y_total = pre_fit_linear(y0, param, weight=True)
    for v in item_list:
        assert_true(v in y_total)

    # no weight pre fit
    x, y_total = pre_fit_linear(y0, param, weight=False)
    for v in item_list:
        assert_true(v in y_total)
