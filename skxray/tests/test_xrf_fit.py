
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import six
import numpy as np
import copy
import logging
from numpy.testing import (assert_equal, assert_array_almost_equal)
from nose.tools import assert_true, raises
from skxray.fitting.base.parameter_data import get_para, e_calibration
from skxray.fitting.xrf_model import (ModelSpectrum, ParamController,
                                      linear_spectrum_fitting,
                                      construct_linear_model, trim,
                                      sum_area, compute_escape_peak,
                                      register_strategy, update_parameter_dict,
                                      _set_parameter_hint, _STRATEGY_REGISTRY)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO,
                    filemode='w')

def synthetic_spectrum():
    param = get_para()
    x = np.arange(2000)
    elemental_lines = ['Ar_K', 'Fe_K', 'Ce_L', 'Pt_M']
    elist, matv, area_v = construct_linear_model(x, param, elemental_lines, default_area=1e5)
    return np.sum(matv, 1) + 100  # avoid zero values


def test_parameter_controller():
    param = get_para()
    elemental_lines = ['Ar_K', 'Fe_K', 'Ce_L', 'Pt_M']
    PC = ParamController(param, elemental_lines)
    set_opt = dict(pos='hi', width='lohi', area='hi', ratio='lo')
    PC.update_element_prop(['Fe_K', 'Ce_L'], **set_opt)
    PC.set_strategy('linear')

    # check boundary value
    for k, v in six.iteritems(PC.params):
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
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO,
                        filemode='w')
    param = get_para()
    elemental_lines = ['Ar_K', 'Fe_K', 'Ce_L', 'Pt_M']
    x0 = np.arange(2000)
    y0 = synthetic_spectrum()

    x, y = trim(x0, y0, 100, 1300)
    MS = ModelSpectrum(param, elemental_lines)
    MS.assemble_models()

    result = MS.model_fit(x, y, weights=1/np.sqrt(y), maxfev=200)
    for k, v in six.iteritems(result.values):
        if 'area' in k:
            # error smaller than 1%
            assert_true((v-1e5)/1e5 < 1e-2)

    # multiple peak sumed, so value should be larger than one peak area 1e5
    sum_Fe = sum_area('Fe_K', result)
    assert_true(sum_Fe > 1e5)

    sum_Ce = sum_area('Ce_L', result)
    assert_true(sum_Ce > 1e5)

    sum_Pt = sum_area('Pt_M', result)
    assert_true(sum_Pt > 1e5)

    # update values
    update_parameter_dict(MS.params, result)
    for k, v in six.iteritems(MS.params):
        if 'area' in MS.params:
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
    x, y_total, area_v = linear_spectrum_fitting(y0, param)
    for v in item_list:
        assert_true(v in y_total)

    # no weight pre fit
    x, y_total, area_v = linear_spectrum_fitting(y0, param, constant_weight=None)
    for v in item_list:
        assert_true(v in y_total)


def test_escape_peak():
    y0 = synthetic_spectrum()
    ratio = 0.01
    param = get_para()
    xnew, ynew = compute_escape_peak(y0, ratio, param)
    # ratio should be the same
    assert_array_almost_equal(np.sum(ynew)/np.sum(y0), ratio, decimal=3)


def test_set_param_hint():
    param = get_para()
    elemental_lines = ['Ar_K', 'Fe_K', 'Ce_L', 'Pt_M']
    bound_options = ['none', 'lohi', 'fixed', 'lo', 'hi']

    MS = ModelSpectrum(param, elemental_lines)
    MS.assemble_models()

    # get compton model
    compton = MS.mod.components[0]

    for v in bound_options:
        input_param = {'bound_type': v, 'max': 13.0, 'min': 9.0, 'value': 11.0}
        _set_parameter_hint('coherent_sct_energy', input_param, compton)
        p = compton.make_params()
        if v == 'fixed':
            assert_equal(p['coherent_sct_energy'].vary, False)
        else:
            assert_equal(p['coherent_sct_energy'].vary, True)
