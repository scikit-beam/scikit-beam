.. _adding_files:

New Sub-packages and Modules
============================

When adding new packages and modules (which map to folders and files)
to the library you are required to update both the build/install
system (:file:`setup.py`) and add files to the documentation folder.

All python source code must be under the main :file:`skbeam` directory.
Non-python sources goes in the :file:`src` directory.

Build
-----

For a folder in the source-tree to be a package it must have an
:file:`__init__.py` file (even if it is empty).  All of the python
(:file:`*.py`) files in that package are then recognized as modules in
that package which can be imported in other files (See Relative Imports header).

In order for :mod:`distutils` to work it must be explicitly told what
packages from the source tree to byte-compile and install.  This is
done via the :code:`packages` key word argument (kwarg) to :func:`setup` in
:file:`setup.py`.  If you add a package, then it's dotted-name must be
added to this list.

e.g. if you add a new package called :code:`utils` to the :code:`skbeam` folder,
the following setup.py file: ::

    setup(
        name='scikit-beam',
        version='0',
        author='Brookhaven National Lab',
        packages=["skbeam"],
        )

    would need to be modified to:

    setup(
        name='scikit-beam',
        version='0',
        author='Brookhaven National Lab',
        packages=["skbeam", "skbeam.utils"],   <------- modification happened here
        )

Documentation
-------------

See :ref:`doc_doc` for documentation about writing and building
the documentation.

Continuing the example from above where a 'utils' source code package was added,
a folder called :file:`/doc/resource/api/utils` should be added.  Let's also
presume that you've got :file:`fitting.py` in the :file:`/skbeam/utils/`.  In the
documentation :file:`/doc/resource/api/utils` folder, create a file named
:file:`index.rst` with the contents: ::

    UTILS API
    =========

    Contents:

    .. toctree::
       :maxdepth: 2

       fitting

Also, add the :file:`/doc/resource/api/utils/index.rst` to
:file:`/doc/resource/api/index.rst`.  This will tell ``sphinx`` to include
the new package in the API documentation.

Now, let's create a module called :file:`fitting.py` in the :file:`utils`
package.  When you add :file:`fitting.py` you need to add a corresponding file
in the documentation folder structure:
:file:`/doc/resource/api/utils/fitting.rst`.  In :file:`fitting.rst` use the
following template: ::

    ======================
     :mod:`fitting` Module
    ======================

    Any prose you want to add about the module, such as examples, discussion,
    or saying hi to your mom can go here.

    .. automodule:: skbeam.core.fitting
       :members:
       :show-inheritance:
       :undoc-members:

This will automatically walk the module to extract and format the doc strings
of all the classes and functions in the module.

Testing
-------

When you add a new module or package please add the corresponding
files and folders in the :file:`skbeam/tests` folder.  Packages get
:file:`test_packagename` and modules get :file:`test_module_name.py`
in the proper directory.

Using the example above, you would create the directory
:file:`/skbeam/tests/test_utils/` and the file :file:`test_fitting.py` in the
:file:`test_utils` folder.

Remember: Write a test for all new functionality!!

Relative Imports
----------------
See the issue (#?) in the scikit-beam repo on github.
