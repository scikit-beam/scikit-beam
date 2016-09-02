.. _documentation-guidelines:

=======================
 Writing Documentation
=======================

High-quality, consistent documentation for the science code is one of
the major goals of the Scikit-beam project.  Hence, we describe our
documentation procedures and rules here.  For the scikit-beam core
project we try to keep to these as closely as possible, while the
standards for affiliated packages are somewhat looser.  (These
procedures and guidelines are still recommended for affiliated
packages, as they encourage useful documentation, a characteristic
often lacking in scientific software.)


Building the Documentation from source
======================================

For information about building the documentation from source, see
the :ref:`builddocs` section in the installation instructions.


Scikit-beam Documentation Rules and Guidelines
==============================================

This section describes the standards for documentation format affiliated
packages that must follow for consideration of integration into the core
module, as well as the standard Scikit-beam docstring format.

* All documentation should be written use the Sphinx documentation tool.

* The template package will provide a recommended general structure for
  documentation. [TODO]

* Docstrings must be provided for all public classes, methods, and functions.

* Docstrings will be incorporated into the documentation using a
  version of numpydoc included with Scikit-beam, and should follow the
  :doc:`docrules`. [TODO do we want to vendor numpydoc or use napolean?]

* Examples and/or tutorials are strongly encouraged for typical use-cases of a
  particular module or class.

* Any external package dependencies aside from NumPy_, SciPy_, or Matplotlib_
  must be explicitly mentioned in the documentation.

* Configuration options using the :mod:`scikit-beam.config` mechanisms must be
  explicitly mentioned in the documentation.  [TODO do we want to keep config]


The details of the docstring format are described on a separate page:

.. toctree::
    docrules



numpydoc Extension
------------------
This extension (and some related extensions) are a port of the
`numpydoc <http://pypi.python.org/pypi/numpydoc/0.3.1>`_ extension
written by the NumPy_ and SciPy_, projects, with some tweaks for
Scikit-beam.  Its main purposes is to reprocess docstrings from code into
a form sphinx understands. Generally, there's no need to interact with
it directly, as docstrings following the :doc:`docrules` will be
processed automatically.



.. _NumPy: http://numpy.scipy.org/
.. _numpydoc: http://pypi.python.org/pypi/numpydoc/0.3.1
.. _Matplotlib: http://matplotlib.sourceforge.net/
.. _SciPy: http://www.scipy.org
.. _Sphinx: http://sphinx.pocoo.org
.. _scikit-beam-helpers: https://github.com/scikit-beam/scikit-beam-helpers
