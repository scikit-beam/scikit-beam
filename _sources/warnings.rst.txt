.. _python-warnings:

**********************
Python warnings system
**********************

.. doctest-skip-all

Scikit-beam uses the Python :mod:`warnings` module to issue warning messages.  The
details of using the warnings module are general to Python, and apply to any
Python software that uses this system.  The user can suppress the warnings
using the python command line argument ``-W"ignore"`` when starting an
interactive python session.  For example::

     $ python -W"ignore"

The user may also use the command line argument when running a python script as
follows::

     $ python -W"ignore" myscript.py

It is also possible to suppress warnings from within a python script.  For
instance, the warnings issued from a single call to the
`scikit-beam.io.fits.writeto` function may be suppressed from within a Python
script using the `warnings.filterwarnings` function as follows::

     >>> import warnings
     >>> from scikit-beam.io import fits
     >>> warnings.filterwarnings('ignore', category=UserWarning, append=True)
     >>> fits.writeto(filename, data, clobber=True)

An equivalent way to insert an entry into the list of warning filter specifications
for simple call `warnings.simplefilter`::

    >>> warnings.simplefilter('ignore', UserWarning)

Scikit-beam includes its own warning classes,
`~scikit-beam.utils.exceptions.Scikit-beamWarning` and
`~scikit-beam.utils.exceptions.Scikit-beamUserWarning`.  All warnings from Scikit-beam are
based on these warning classes (see below for the distinction between them). One
can thus ignore all warnings from Scikit-beam (while still allowing through
warnings from other libraries like Numpy) by using something like::

    >>> from scikit-beam.utils.exceptions import Scikit-beamWarning
    >>> warnings.simplefilter('ignore', category=Scikit-beamWarning)

Warning filters may also be modified just within a certain context using the
`warnings.catch_warnings` context manager::

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter('ignore', Scikit-beamWarning)
    ...     fits.writeto(filename, data, clobber=True)

As mentioned above, there are actually *two* base classes for Scikit-beam warnings.
The main distinction is that `~scikit-beam.utils.exceptions.Scikit-beamUserWarning` is
for warnings that are *intended* for typical users (e.g. "Warning: Ambiguous
unit", something that might be because of improper input).  In contrast,
`~scikit-beam.utils.exceptions.Scikit-beamWarning` warnings that are *not*
`~scikit-beam.utils.exceptions.Scikit-beamUserWarning` may be for lower-level warnings
more useful for developers writing code that *uses* Scikit-beam (e.g., the
deprecation warnings discussed below).  So if you're a user that just wants to
silence everything, the code above will suffice, but if you are a developer and
want to hide development-related warnings from your users, you may wish to still
allow through `~scikit-beam.utils.exceptions.Scikit-beamUserWarning`.

Scikit-beam also issues warnings when deprecated API features are used.  If you
wish to *squelch* deprecation warnings, you can start Python with
``-Wi::Deprecation``.  This sets all deprecation warnings to ignored.  There is
also an Scikit-beam-specific `~scikit-beam.utils.exceptions.Scikit-beamDeprecationWarning`
which can be used to disable deprecation warnings from Scikit-beam only.

See `the CPython documentation
<http://docs.python.org/2/using/cmdline.html#cmdoption-W>`__ for more
information on the -W argument.
