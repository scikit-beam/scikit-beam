==========================================
 Building Scikit-beam and its Subpackages
==========================================

.. warning::

   This page still needs to be adapted from astropy

The build process currently uses the `setuptools
<https://bitbucket.org/pypa/setuptools>`_ package to build and install the
scikit-beam core (and any affiliated packages that use the template).


Customizing setup/build for subpackages
=======================================

As is typical, there is a single ``setup.py`` file that is used for the whole
``scikit-beam`` package.  To customize setup parameters for a given sub-package, a
``setup_package.py`` file can be defined inside a package, and if it is present,
the setup process will look for the following functions to customize the build
process:

* ``get_package_data``
    This function, if defined, should return a dictionary mapping the name of
    the subpackage(s) that need package data to a list of data file paths
    (possibly including wildcards) relative to the path of the package's source
    code.  e.g. if the source distribution has a needed data file
    ``scikit-beam/wcs/tests/data/3d_cd.hdr``, this function should return
    ``{'scikit-beam.wcs.tests':['data/3d_cd.hdr']}``. See the ``package_data``
    option of the  :func:`distutils.core.setup` function.

    It is recommended that all such data be in a directory named ``data`` inside
    the package within which it is supposed to be used.  This package data should
    be accessed via the ``scikit-beam.utils.data.get_pkg_data_filename`` and
    ``scikit-beam.utils.data.get_pkg_data_fileobj`` functions.

* ``get_extensions``
    This provides information for building C or Cython extensions. If defined,
    it should return a list of :class:`distutils.core.Extension` objects controlling
    the Cython/C build process (see below for more detail).

* ``get_build_options``
    This function allows a package to add extra build options.  It
    should return a list of tuples, where each element has:

    - *name*: The name of the option as it would appear on the
      commandline or in the ``setup.cfg`` file.

    - *doc*: A short doc string for the option, displayed by
      ``setup.py build --help``.

    - *is_bool* (optional): When `True`, the option is a boolean
      option and doesn't have an associated value.

* ``get_external_libraries``
    This function declares that the package uses libraries that are
    included in the scikit-beam distribution that may also be distributed
    elsewhere on the users system.  It should return a list of library
    names.  For each library, a new build option is created,
    ``'--use-system-X'`` which allows the user to request to use the
    system's copy of the library.
