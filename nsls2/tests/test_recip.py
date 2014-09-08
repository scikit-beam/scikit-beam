from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import nsls2.recip as recip
import numpy.testing as npt
import six
from nsls2.testing.decorators import known_fail_if


@known_fail_if(six.PY3)
def test_process_to_q():
    detector_size = (256, 256)
    pixel_size = (0.0135*8, 0.0135*8)
    calibrated_center = (256/2.0, 256/2.0)
    dist_sample = 355.0

    energy = 640  # (  in eV)
    # HC_OVER_E to convert from Energy to wavelength (Lambda)
    hc_over_e = 12398.4
    wavelength = hc_over_e / energy  # (Angstrom )

    ub_mat = np.array([[-0.01231028454, 0.7405370482, 0.06323870032],
                       [0.4450897473, 0.04166852402, -0.9509449389],
                       [-0.7449130975, 0.01265920962, -0.5692399963]])

    setting_angles = np.array([[40., 15., 30., 25., 10., 5.],
                              [90., 60., 0., 30., 10., 5.]])
    # delta=40, theta=15, chi = 90, phi = 30, mu = 10.0, gamma=5.0

    hkl_val = recip.process_to_q(setting_angles, detector_size,
                                 pixel_size, calibrated_center,
                                dist_sample, wavelength, ub_mat)

    # Known HKL values for the given six angles)
    # each entry in list is (pixel_number, known hkl value)
    known_hkl = [(32896, np.array([-0.15471196, 0.19673939, -0.11440936])),
                 (98432, np.array([0.10205953,  0.45624416, -0.27200778]))]

    for pixel, kn_hkl in known_hkl:
        npt.assert_array_almost_equal(hkl_val[pixel], kn_hkl, decimal=8)


@known_fail_if(six.PY3)
def test_process_grid():
    size = 10
    q_max = np.array([1.0, 1.0, 1.0])
    q_min = np.array([-1.0, -1.0, -1.0])
    dqn = np.array([size, size, size])
    # slice tricks
    # this make a list of slices, the imaginary value in the
    # step is interpreted as meaning 'this many values'
    slc = [slice(_min + (_max - _min)/(s * 2),
                 _max - (_max - _min)/(s * 2),
                 1j * s)
           for _min, _max, s in zip(q_min, q_max, dqn)]
    # use the numpy slice magic to make X, Y, Z these are dense meshes with
    # points in the center of each bin
    X, Y, Z = np.mgrid[slc]

    # make and ravel the image data (which is all ones)
    I = np.ones_like(X).ravel()

    # make input data (Nx3
    data = np.array([np.ravel(X),
                     np.ravel(Y),
                     np.ravel(Z)]).T

    (grid_data, grid_occu,
         grid_std, grid_out) = recip.process_grid(data, I,
                                                  q_min, q_max,
                                                  dqn=dqn)

    # check the values are as expected
    npt.assert_array_equal(grid_data.ravel(), I)
    npt.assert_equal(grid_out, 0)
    npt.assert_array_equal(grid_occu, np.ones_like(grid_occu))
    npt.assert_array_equal(grid_std, 0)


@known_fail_if(six.PY3)
def test_process_grid_std():
    size = 10
    q_max = np.array([1.0, 1.0, 1.0])
    q_min = np.array([-1.0, -1.0, -1.0])
    dqn = np.array([size, size, size])
    # slice tricks
    # this make a list of slices, the imaginary value in the
    # step is interpreted as meaning 'this many values'
    slc = [slice(_min + (_max - _min)/(s * 2),
                 _max - (_max - _min)/(s * 2),
                 1j * s)
           for _min, _max, s in zip(q_min, q_max, dqn)]
    # use the numpy slice magic to make X, Y, Z these are dense meshes with
    # points in the center of each bin
    X, Y, Z = np.mgrid[slc]

    # make and ravel the image data (which is all ones)
    I = np.hstack([j * np.ones_like(X).ravel() for j in range(1, 6)])

    # make input data (N*5x3)
    data = np.vstack([np.tile(_, 5)
                      for _ in (np.ravel(X), np.ravel(Y), np.ravel(Z))]).T

    (grid_data, grid_occu,
         grid_std, grid_out) = recip.process_grid(data, I,
                                                  q_min, q_max,
                                                  dqn=dqn)

    # check the values are as expected
    npt.assert_array_equal(grid_data,
                           np.ones_like(X) * np.mean(np.arange(1, 6)))
    npt.assert_equal(grid_out, 0)
    npt.assert_array_equal(grid_occu, np.ones_like(grid_occu)*5)
    # need to convert std -> ste (standard error)
    # according to wikipedia ste = std/sqrt(n), but experimentally, this is
    # implemented as ste = std / srt(n - 1)
    npt.assert_array_equal(grid_std,
                           (np.ones_like(grid_occu) *
                            np.std(np.arange(1, 6))/np.sqrt(5 - 1)))


def test_convert_to_q_saxs():
   detector_size = (2048, 2048)
   pixel_size = (0.2, 0.2) # (mm)
   calibrated_center = (1006.58, 1043.71)  # (mm)
   dist_sample = 3500.00  # (mm)
   wavelength = 0.1859  # (Angstrom )

   q_values = recip.convert_to_q_saxs(detector_size, pixel_size, dist_sample,
                                      calibrated_center, wavelength)

def test_convert_to_q_waxs():
   detector_size = (2048, 2048)
   pixel_size = (0.2, 0.2) # (mm)
   calibrated_center = (1006.58, 1043.71)  # (mm)
   dist_sample = 1498.76 # (mm)
   wavelength = 0.1859  # (Angstrom )

   q_values = recip.convert_to_q_waxs(detector_size,pixel_size, dist_sample,
                                      calibrated_center, wavelength)

def test_convert_to_q_gisaxs():
   detector_size = (256, 256)
   pixel_size = (0.0135*8, 0.0135*8)
   calibrated_center = (256/2.0, 256/2.0)
   dist_sample = 355.0
   ref_beam = (143, 123)
   incident_angle =
   rod_geometry = None

   energy = 640  # (  in eV)
   # HC_OVER_E to convert from Energy to wavelength (Lambda)
   hc_over_e = 12398.4
   wavelength = hc_over_e / energy  # (Angstrom )

   q_values = recip.convert_to_q_waxs_this(detector_size, pixel_size, dist_sample,
                      calibrated_center, wavelength)