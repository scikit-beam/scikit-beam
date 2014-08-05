.. _adding_files:

New Sub-packages and Modules
============================

When adding new packages and modules (which map to folders and files)
to the library.  Doing so requires updating both the build/install
system (:file:`setup.py`) and adding files to the documentation folder.

All source-files must be under the main :file:`nsls2` directory.

Build
-----

For a folder in the source-tree to be a package it must have a
:file:`__init__.py` file (even if it is empty).  All of the python
(:file:`*.py`) files in that package are then recognized as modules in
that package.

In order for :mod:`disutils` to work it must be explicitly told what
packages from the source tree to byte-compile and install.  This is
done via the :code:`packages` kwarg to :func:`setup` in
:file:`setup.py`.  If you add a package, then it's dotted-name must be
added to this list.

Documentation
-------------

See :ref:`doc_doc` for documentation about writing and building
the documentation.

When adding a new package create a corresponding folder in
:file:`doc/rsources/api`.  In that folder create a file :file:`index.rst`
with the contens::


    Package name
    ============

    Contents:

    .. toctree::
       :maxdepth: 2



and add the following line ::

    new_package/index

to the :rst:role:`toctree` of the :file:`index.rst` in the parent directory.
This will tell :prog:`sphinx` to include the new


When creating a new module create a new module add a corresponding file following
this template: ::

    =========================
     :mod:`new_module` Module
    =========================

    Any prose you want to add about the module, such as examples, discussion,
    or saying hi to your mom can go here.

    .. automodule:: doted.path.new_module
       :members:
       :show-inheritance:
       :undoc-members:

This will automatically walk the module to extract and format the doc strings
of all the classes and functions in the module.

Testing
-------

When you add a new module or package please add the corresponding
files and folders in the :file:`nsls2/tests` folder.  Packages get
:file:`test_packagename` and modules get :file:`test_module_name.py`
in the proper directory.
