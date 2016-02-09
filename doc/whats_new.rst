.. _whats_new:

What's new
**********

A catalog of new features, improvements, and bug-fixes in each release.
Follow links to the relevant GitHub issue or pull request for specific
code changes and any related discussion.

v0.0.8
------
In-progress
===========
- `PR #395 <https://github.com/scikit-beam/scikit-beam/pull/395>`_: One-time from two-time
- `PR #401 <https://github.com/scikit-beam/scikit-beam/pull/401>`_: Removing bad images from xsvs
- `PR #403 <https://github.com/scikit-beam/scikit-beam/pull/401>`_: Adding four time correlation
- `PR #405 <https://github.com/scikit-beam/scikit-beam/pull/405>`_: Implement user peak, so users have more flexible control of escape peak or pileup peaks.
- `PR #373 <https://github.com/scikit-beam/scikit-beam/pull/373>`_: Major overhaul of the docs

v0.0.7
------
- Added multi-tau scheme two-time correlation. `PR #391 <https://github.com/scikit-beam/scikit-beam/pull/391>`_
  This two-time correlator can be found at `skbeam.core.correlation:two_time_corr`.
  There is a generator version that returns its internal state after consuming
  each image, `skbeam.core.correlation:lazy_two_time`. The value that the
  generator yields should be passed to `skbeam.core.correlation:two_time_state_to_results`
  to produce the correlation results and the lag steps that the correlation results
  correspond to.
- `PR #400 <https://github.com/scikit-beam/scikit-beam/pull/400>`_ Added
  removing bad images from multi-tau one time correlation.

  - Added new module `skbeam.core.mask`.
    It contains functions specific to mask or threshold an image
    basically to clean images. This module contains:

    - Convert the images marked as "bad" in bad list by their index in images into
      a np.nan array.
    - Set all pixels whose value is greater than `threshold` to 0 and yields the
      thresholded images out.


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
