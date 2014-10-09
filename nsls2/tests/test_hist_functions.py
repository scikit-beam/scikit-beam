
#TODO: Need to sort out tests for each function and operation as a whole.
import numpy as np
import nsls2.img_proc.histops as histops
from histops import rescale_intensity_values as rscale
from nsls2.core import bin_1D
from nsls2.core import bin_edges_to_centers
from nose.tools import assert_raises
from numpy.testing import assert_array_almost_equal

%load_ext autoreload
%autoreload 2

#Test case number 1: float values ranging from 0 to 1
test_1 = np.zeros(7)
test_1[0:5] = np.random.rand(5)
test_1[6] = 1.0
test_2 = np.array([0.10549548,
                         0.18549702,
                         0.64457573,
                         0.89678006,
                         0.37372467,
                         0.,
                         1.,
                         0.92712985,
                         0.06273057,
                         0.98712985])
test_3 = test_2 * 10000
test_4 = (test_2 - 0.3) * 12000
test_5 = np.floor(test_2*10)
#Large test arrays
test_5 = np.random.rand(5,5,5)


def get_base_test_array():
    test_array = np.random.rand(5,5,5)
    return test_array


def test_rescale_intensity_values_flt2int():
    base_test_array = get_base_test_array()
    small_flt_test = base_test_array
    lrg_flt_test = base_test_array*10000
    end_ranges = [(-5.0,5.0), (-32000.0,32000.0), (0,1)]
    initial_values_list = [small_flt_test, lrg_flt_test]
    for i in initial_values_list:
        for f in end_ranges:
            rescale_1 = histops.rescale_intensity_values(i,
                                                         new_max=np.amax(f),
                                                         new_min=np.amin(f),
                                                         out_dType=f.dtype)
            rescale_2 = histops.rescale_intensity_values(rescale_1,
                                                         new_max=np.amax(i),
                                                         new_min=np.amin(i),
                                                         out_dType=i.dtype)
            assert_equals(rescale_2 == i)


def test_rescale_intensity_values_int2int():
    base_test_array = get_base_test_array()
    small_int_test = np.floor(base_test_array*5)
    med_int_test = np.floor(base_test_array*255)
    lrg_int_test = np.floor(base_test_array*10000)
    end_ranges = [(0,3), (1,10), (-5000, 12000), (0,255)]
    histops.rescale_intensity_values


def test_rescale_intensity_values_int2flt():
    base_test_array = get_base_test_array()
    small_int_test = np.floor(base_test_array*5)
    med_int_test = np.floor(base_test_array*255)
    lrg_int_test = np.floor(base_test_array*10000)
    end_ranges = [(-5.0,5.0), (-32000.0,32000.0), (0,1)]
    histops.rescale_intensity_values


def gauss_crv_make(height,
                   center,
                   width,
                   num_values,
                   x_min,
                   x_max,
                   limit_value=0):
    height = float(height)
    center = float(center)
    width = float(width)
    x_min = float(x_min)
    x_max = float(x_max)
    x_values = np.arange(x_min, x_max, ((x_max-x_min)/(num_values)))
    y_values = height*np.exp(-((x_values-center)**2)/(2*width**2))+limit_value
    return (x_values, y_values)


def test_hist_make():
    h1 = 500
    cen1 = 0
    w1 = 400
    num_values = 3000
    x_min1 = -2400
    x_max1 = 2400

    h2 = 350
    cen2 = 4500
    w2 = 800
    x_min2 = 0
    x_max2 = 9000

    x_crv_1, y_crv_1 = gauss_crv_make(h1,
                                      cen1,
                                      w1,
                                      num_values,
                                      x_min1,
                                      x_max1)
    x_crv_2, y_crv_2 = gauss_crv_make(h2,
                                      cen2,
                                      w2,
                                      num_values,
                                      x_min2,
                                      x_max2)
    merged_crvs = np.array(zip(np.concatenate((x_crv_1, x_crv_2)),
                               np.concatenate((y_crv_1, y_crv_2))))
    hist_src_synth = merged_crvs[merged_crvs[:,0].argsort()]
    nx = (np.amax((x_min1, x_max1, x_min2, x_max2)) -
          np.amin((x_min1, x_max1, x_min2, x_max2)))
    bin_edges, vals, count = bin_1D(hist_src_synth[:,0],
                                    hist_src_synth[:,1], nx=nx)
    vals = np.floor(vals)
    bin_avg = bin_edges_to_centers(bin_edges)
    cell_location = 0
    vals_sum = 0
    synth_data=np.empty(vals.sum())
    for _ in np.arange(len(vals)):
        for counts in np.arange(vals[_]):
            synth_data[cell_location] = bin_avg[_]
            cell_location+=1

    hist, edges, avg = histops.hist_make(synth_data, len(vals))
    source_curve = np.array(zip(bin_avg, vals))
    hist_make_results = np.array(zip(avg, hist))
    area_s_curve = source_curve[:,1].sum()
    area_hist_make = hist_make_results[:,1].sum()
    assert_array_almost_equal(area_s_curve,
                              area_hist_make,
                              decimal=3,
                              err_msg=('hist_make results and synthetic curve'\
                                       'do not match. Area under synthetic '\
                                       'curve equals {0}, area under the '\
                                       'histogram generated using hist_make '\
                                       'equals: {1}'.format(area_s_curve,
                                                            area_hist_make,)))

