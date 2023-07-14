:tocdepth: 2

==================
 Scikit-beam Core
==================


The :ref:`vision<vision>` of ``scikit-beam`` is to provide simple
functions useful for the X-ray, Neutron and Electron communities.  The
primary goal of the scikit-beam project is to provide a centralized
repository for algorithms that are used in these three scientific
domains.  ``scikit-beam`` functions accept and return standard Python
and numpy datatypes, so they integrate well with other packages from
the scientific Python community.  Further, the modular design of
scikit-beam allows its components to be easily reused in ways not
envisioned by the authors.



Scikit-beam is being developed at the National Synchrotron Light
Source II at Brookhaven National Lab and also in collaboration with
scientists at the LCLS-II and APS.


Supported techniques
====================

* Differential Phase Contrast (:mod:`~skbeam.core.dpc`)
* CDI (:mod:`~skbeam.core.cdi`)
* MultiTau correlation (:mod:`~skbeam.core.correlation`)
* X-Ray Speckle Visibility Spectroscopy (XSVS) (:mod:`~skbeam.core.speckle`)
* X-ray Fluorescence  (:mod:`~skbeam.fluorescence`)
* Fast histograms

Other utilities
===============

* Basic constants (:mod:`~skbeam.core.constants`)
* Logical convenience functions (:mod:`~skbeam.core.arithmetic`)
* Utilities to estimating the center of a ring pattern and the
  sample-to-detector distance of a powder pattern
  (:mod:`~skbeam.core.calibration`)
* Peak extraction (:mod:`~skbeam.core.feature`)
* Mask pixels based on a threshold; as a statistical outlier within a bin; for
  proximity to canvas edge (margin). (:mod:`~skbeam.core.mask`)
* Compute reciprocol space coordinates of pixels. (:mod:`~skbeam.core.recip`)
* Draw and manipulate ROI mask; draw kymograph; compute statistics on ROIs.
  (:mod:`~skbeam.core.roi`)
* Misc. utilities (:mod:`~skbeam.core.utils`)
* A thin wrapper around ``scipy.stats.binned_statistic``
  (:mod:`~skbeam.core.stats`)

.. _installation_tl:

Installation
============

.. toctree::
   :maxdepth: 1

   install
   getting_started
   installation


.. _reporting_issues:

Reporting Issues
================

If you have found a bug in scikit-beam please report it. The preferred
way is to create a new issue on the scikit-beam `GitHub issue page
<http://github.com/scikit-beam/scikit-beam/issues>`_; that requires
`creating a free account <https://github.com>`_ on GitHub if you do
not have one.

Please include an example that demonstrates the issue that will allow
the developers to reproduce and fix the problem. You may be asked to
also provide information about your operating system and a full Python
stack trace; the Skbeam developers will walk you through obtaining a
stack trace if it is necessary.

API Docs
--------



.. toctree::
   :maxdepth: 3

   resource/api/index

.. _contributing:


Contributing
============

The scikit-beam project is made both by and for its users, so we
highly encourage contributions at all levels.  This spans the gamut
from sending an email mentioning a typo in the documentation or
requesting a new feature all the way to developing a major new
package.

The full range of ways to be part of the Skbeam project are described
at `Contribute to scikit-beam
<http://scikit-beam.github.io/contribute.html>`_. To get started
contributing code or documentation (no git or GitHub experience
necessary):

.. toctree::
    :maxdepth: 1

    development/workflow/get_devel_version
    development/workflow/development_workflow


.. _developer-docs:

Developer Documentation
=======================

The developer documentation contains instructions for how to contribute to
Skbeam or affiliated packages, as well as coding, documentation, and
testing guidelines. For the guiding vision of this process and the project
as a whole, see :doc:`development/vision`.

.. toctree::
   :maxdepth: 1

   overview
   development/workflow/development_workflow
   development/codeguide
   development/docguide
   development/testguide
   development/scripts
   development/building
   development/ccython
   development/releasing
   development/workflow/maintainer_workflow
   development/affiliated-packages
   development/vision
   resource/dev_guide/index
   api_changes
   python_versions
   warnings

other
=====
.. toctree::
   :maxdepth: 1


.. toctree::
   :maxdepth: 1

   whatsnew/index
   whats_new
   known_issues
   generated/examples/index
   introduction

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
