
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import six
import numpy as np
import copy
from numpy.testing import (assert_equal, assert_array_almost_equal)
from nose.tools import assert_true, raises
from skxray.fitting.base.parameter_data import get_para, e_calibration, adjust_element
from skxray.fitting.xrf_model import (ModelSpectrum, ParamController,
                                      pre_fit_linear,
                                      get_linear_model, set_range,
                                      get_sum_area, get_escape_peak,
                                      register_strategy, update_parameter_dict,
                                      _STRATEGY_REGISTRY)


def synthetic_spectrum():
    param = get_para()
    x = np.arange(2000)

    elist, matv = get_linear_model(x, param, default_area=1e5)
    return np.sum(matv, 1) + 100  # avoid zero values


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

    # multiple peak sumed, so value should be larger than one peak area 1e5
    sum_Fe = get_sum_area('Fe', result)
    assert_true(sum_Fe > 1e5)

    sum_Ce = get_sum_area('Ce_L', result)
    assert_true(sum_Ce > 1e5)

    sum_Pt = get_sum_area('Pt_M', result)
    assert_true(sum_Ce > 1e5)

    # update values
    update_parameter_dict(MS.parameter, result)
    for k, v in six.iteritems(MS.parameter):
        if 'area' in MS.parameter:
            assert_equal(v['value'], result.values[k])


def test_register():
    new_strategy = e_calibration
    register_strategy('e_calibration', new_strategy, overwrite=False)
    assert_equal(len(_STRATEGY_REGISTRY), 5)

    new_strategy = copy.deepcopy(e_calibration)
    new_strategy['coherent_sct_amplitude'] = 'fixed'
    register_strategy('new_strategy', new_strategy)
    assert_equal(len(_STRATEGY_REGISTRY), 6)


@raises(RuntimeError)
def test_register_error():
    new_strategy = copy.deepcopy(e_calibration)
    new_strategy['coherent_sct_amplitude'] = 'fixed'
    register_strategy('e_calibration', new_strategy, overwrite=False)


def test_pre_fit():
    y0 = synthetic_spectrum()

    # the following items should appear
    item_list = ['Ar_K', 'Fe_K', 'compton', 'elastic']

    param = get_para()

    # with weight pre fit
    x, y_total = pre_fit_linear(y0, param, weight=True)
    for v in item_list:
        assert_true(v in y_total)

    for k, v in six.iteritems(y_total):
        print(k)
    # no weight pre fit
    x, y_total = pre_fit_linear(y0, param, weight=False)
    for v in item_list:
        assert_true(v in y_total)



def test_escape_peak():
    y0 = synthetic_spectrum()
    ratio = 0.01
    param = get_para()
    xnew, ynew = get_escape_peak(y0, ratio, param)
    # ratio should be the same
    assert_array_almost_equal(np.sum(ynew)/np.sum(y0), ratio, decimal=3)


