
#TODO: Need to sort out tests for each function and operation as a whole.
import numpy as np
import nsls2.img_proc.histops as histops
from histops import rescale_intensity_values as rscale
from nsls2.core import bin_1D
from nsls2.core import bin_edges_to_centers
from nose.tools import assert_raises
from numpy.testing import assert_array_almost_equal


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

