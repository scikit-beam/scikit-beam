.. _whats_new:

What's new
**********

A catalog of new features, improvements, and bug-fixes in each release.
Follow links to the relevant GitHub issue or pull request for specific
code changes and any related discussion.


v0.0.7 (unreleased)
-------------------
- Added multi-tau scheme two-time correlation. `PR #391 <https://github.com/scikit-beam/scikit-beam/pull/391>`_
  This two-time correlator can be found at `skbeam.core.correlation:two_time_corr`.
  There is a generator version that returns its internal state after consuming
  each image, `skbeam.core.correlation:lazy_two_time`. The value that the
  generator yields should be passed to `skbeam.core.correlation:two_time_state_to_results`
  to produce the correlation results and the lag steps that the correlation results
  correspond to.
- Added removing bad images from multi-tau one time correlation.
  `PR #400 <https://github.com/scikit-beam/scikit-beam/pull/400>`_
  This multi-tau one time correlator can be found at `skbeam.core.correlation:multi_tau_auto_corr`.
- Added new module `skbeam.core.mask`. `PR #400 <https://github.com/scikit-beam/scikit-beam/pull/400>`_
  It contains functions specific to mask or threshold an image
  basically to clean images. Added following two functions to this module.
  * Added a generator `skbeam.core.mask.bad_to_nan' to convert the images marked as "bad" in bad
    list by their index in images into a np.nan array
  * Added a generator `skbeam.core.mask.threshold_mask` to set all pixels whose value is greater
    than `threshold` to 0 and yields the thresholded images out.


v0.0.6
------
- Partial data, generator implementation of one-time-correlation
- Rename from scikit-xray to scikit-beam and skxray->skbeam
- Add cython implementation of accumulating histograms
- Clean up the ctrans code
- Add multiprocessing single pixel fitting
- Automatically build the docs on traivs


v0.0.5
------

New Functionality
=================
* X-Ray Speckle Visibility Spectroscopy `PR 293 <https://github.com/scikit-beam/scikit-beam/pull/293>`_
* Fitting 1-time correlation data to ISF equation, `PR 295 <https://github.com/scikit-beam/scikit-beam/pull/295>`_
* Kymograph (aka waterfall plot), `PR  306 <https://github.com/scikit-beam/scikit-beam/pull/306>`_


API Changes
===========
* :func:`weighted_nnls_fit` was removed from :mod:`skbeam.core.fitting.xrf_model`.
  Weighted nnls fitting was combined into :func:`nnls_fit`, which includes
  weights as a new argument.

* :func:`extract_label_indices` is a helper function for labeled arrays and
  was moved to its new home in `skbeam.core.roi` from `skbeam.core.correlation`

Other updates
=============
* `PR 316 <https://github.com/scikit-beam/scikit-beam/pull/316>`_: Do a better
  job isolating dependencies so that our "optional" packages truly are optional
* `PR 319 <https://github.com/scikit-beam/scikit-beam/pull/319>`_: Use latest
  lmfit version published to scikit-beam anaconda.org channel in travis build
* `PR 326 <https://github.com/scikit-beam/scikit-beam/pull/326>`_:
  Add quick start guide and note about testing
* `PR 327 <https://github.com/scikit-beam/scikit-beam/pull/327>`_: Pin to lmfit
  0.8.3 in conda recipe
* `PR 332 <https://github.com/scikit-beam/scikit-beam/pull/332>`_: Correct the
  equation in the one-time correlation docstring
* `PR 333 <https://github.com/scikit-beam/scikit-beam/pull/333>`_: Update
  readme with new examples in `scikit-beam-examples <https://github.com/scikit-beam/scikit-beam-examples>`_
