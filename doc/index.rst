.. Skbeam documentation master file, created by
   sphinx-quickstart on Tue Jul 26 02:59:34 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:tocdepth: 2

######################################
Scikit-beam Core Package Documentation
######################################

Welcome to the scikit-beam documentation! Scikit-beam is a
community-driven package intended to contain much of the core
functionality and some common tools needed for performing x-ray,
electron, and neutron science with Python.

.. _user-docs:

******************
User Documentation
******************

**Skbeam at a glance**

.. toctree::
   :maxdepth: 1

   overview
   install
   getting_started

**Core data structures and transformations**

TODO

**Connecting up: Files and I/O**

TODO

**Nuts and bolts of Skbeam**

Documentation
-------------
.. toctree::
   :maxdepth: 1

   logging
   warnings


**Skbeam project details**

.. toctree::
   :maxdepth: 1

   stability
   whatsnew/index
   known_issues
   credits
   license

.. _getting_help:

************
Getting help
************

TBD

.. _reporting_issues:

****************
Reporting Issues
****************

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



************
Contributing
************

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

***********************
Developer Documentation
***********************

The developer documentation contains instructions for how to contribute to
Skbeam or affiliated packages, as well as coding, documentation, and
testing guidelines. For the guiding vision of this process and the project
as a whole, see :doc:`development/vision`.

.. toctree::
   :maxdepth: 1

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
   changelog

******************
Indices and Tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
