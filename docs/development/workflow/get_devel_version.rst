.. _get_devel:

===========================
Try the development version
===========================

.. note::
    `git`_ is the name of a source code management system. It is used to keep
    track of changes made to code and to manage contributions coming from
    several different people. If you want to read more about `git`_ right now
    take a look at `Git Basics`_.

    If you have never used `git`_ before, allow one hour the first time you do
    this. You will not need to do this every time you want to contribute;
    most of it is one-time setup. You can count on one hand the number of
    `git`_ commands you will need to regularly use to keep your local copy
    of `Scikit-beam`_ up to date. If you find this taking more than an hour email
    the scikit-beam developers.


Trying out the development version of Scikit-beam is useful in three ways:

* More users testing new features helps uncover bugs before the feature is
  released.
* A bug in the most recent stable release might have been fixed in the
  development version. Knowing whether that is the case can make your bug
  reports more useful.
* You will need to go through all of these steps before contributing any
  code to Scikit-beam. Practicing now will save you time later if you plan to
  contribute.

Overview
--------

Conceptually, there are several steps to getting a working copy of the latest
version of Scikit-beam on your computer:

#. :ref:`fork_a_copy`; this copy is called a *fork* (if you don't have an
   account on `github`_ yet, go there now and make one).
#. :ref:`check_git_install`
#. :ref:`clone_your_fork`; this is called making a *clone* of the repository.
#. :ref:`set_upstream_master`
#. :ref:`make_a_branch`; this is called making a *branch*.
#. :ref:`activate_development_scikit-beam`
#. :ref:`test_installation`
#. :ref:`try_devel`


Step-by-step instructions
-------------------------

.. _fork_a_copy:

Make your own copy of Scikit-beam on GitHub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the language of `GitHub`_, making a copy of someone's code is called making
a *fork*. A fork is a complete copy of the code and all of its revision
history.

#. Log into your `GitHub`_ account.

#. Go to the `Scikit-beam GitHub`_ home page.

#. Click on the *fork* button:

   .. image:: ../workflow/forking_button.png

   After a short pause and an animation of Octocat scanning a book on a
   flatbed scanner, you should find yourself at the home page for your own
   forked copy of Scikit-beam_.

.. _check_git_install:

Make sure git is installed and configured on your computer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Check that git is installed:**

Check by typing, in a terminal::

    $ git --version
    # if git is installed, will get something like: git version 1.9.1

If `git`_ is not installed, `get it <http://git-scm.com/downloads>`_.

**Basic git configuration:**

Follow the instructions at `Set Up Git at GitHub`_ to take care of two
essential items:

+ Set your user name and email in your copy of `git`_

+ Set up authentication so you don't have to type your github password every
  time you need to access github from the command line. The default method at
  `Set Up Git at GitHub`_ may require administrative privileges; if that is a
  problem, set up authentication
  `using SSH keys instead <https://help.github.com/articles/generating-ssh-keys>`_

We also recommend setting up `git`_ so that when you copy changes from your
computer to `GitHub`_ only the copy (called a *branch*) of Scikit-beam that you are
working on gets pushed up to GitHub.  *If* your version of git is 1.7.11 or,
greater, you can do that with::

    git config --global push.default simple

If you skip this step now it is not a problem; `git`_ will remind you to do it in
those cases when it is relevant.  If your version of git is less than 1.7.11,
you can still continue without this, but it may lead to confusion later, as you
might push up branches you do not intend to push.

.. note::

    Make sure you make a note of which authentication method you set up
    because it affects the command you use to copy your GitHub fork to your
    computer.

    If you set up password caching (the default method) the URLs will look like
    ``https://github.com/your-user-name/scikit-beam.git``.

    If you set up SSH keys the URLs you use for making copies will look
    something like ``git@github.com:your-user-name/scikit-beam.git``.


.. _clone_your_fork:

Copy your fork of Scikit-beam from GitHub to your computer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One of the commands below will make a complete copy of your `GitHub`_ fork
of `Scikit-beam`_ in a directory called ``scikit-beam``; which form you use depends
on what kind of authentication you set up in the previous step::

    # Use this form if you setup SSH keys...
    $ git clone git@github.com:your-user-name/scikit-beam.git
    # ...otherwise use this form:
    $ git clone https://github.com/your-user-name/scikit-beam.git

If there is an error at this stage it is probably an error in setting up
authentication.

.. _set_upstream_master:

Tell git where to look for changes in the development version of Scikit-beam
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Right now your local copy of `Scikit-beam`_ doesn't know where the development
version of `Scikit-beam`_ is. There is no easy way to keep your local copy up to
date. In `git`_ the name for another location of the same repository is a
*remote*. The repository that contains the latest "official" development
version is traditionally called the *upstream* remote, but here we use a
more meaningful name for the remote: *scikit-beam*.

Change into the ``scikit-beam`` directory you created in the previous step and
let `git`_ know about about the scikit-beam remote::

    cd scikit-beam
    git remote add scikit-beam git://github.com/scikit-beam/scikit-beam.git

You can check that everything is set up properly so far by asking `git`_ to
show you all of the remotes it knows about for your local repository of
`Scikit-beam`_ with ``git remote -v``, which should display something like::

    scikit-beam   git://github.com/scikit-beam/scikit-beam.git (fetch)
    scikit-beam   git://github.com/scikit-beam/scikit-beam.git (push)
    origin     git@github.com:your-user-name/scikit-beam.git (fetch)
    origin     git@github.com:your-user-name/scikit-beam.git (push)

Note that `git`_ already knew about one remote, called *origin*; that is your
fork of Scikit-beam on `GitHub`_.

To make more explicit that origin is really *your* fork of Scikit-beam, rename that
remote to your `GitHub`_ user name::

  git remote rename origin your-user-name

.. _make_a_branch:

Create your own private workspace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One of the nice things about `git`_ is that it is easy to make what is
essentially your own private workspace to try out coding ideas. `git`_
calls these workspaces *branches*.

Your repository already has several branches; see them if you want by running
``git branch -a``. Most of them are on ``remotes/origin``; in other words,
they exist on your remote copy of Scikit-beam on GitHub.

There is one special branch, called *master*. Right now it is the one you are
working on; you can tell because it has a marker next to it in your list of
branches: ``* master``.

To make a long story short, you never want to work on master. Always work on a branch.

To avoid potential confusion down the road, make your own branch now; this
one you can call anything you like (when making contributions you should use
a meaningful more name)::

    git branch my-own-scikit-beam

You are *not quite* done yet. Git knows about this new branch; run
``git branch`` and you get::

    * master
      my-own-scikit-beam

The ``*`` indicates you are still working on master. To work on your branch
instead you need to *check out* the branch ``my-own-scikit-beam``. Do that with::

    git checkout my-own-scikit-beam

and you should be rewarded with::

    Switched to branch 'my-own-scikit-beam'

.. _activate_development_scikit-beam:

"Activate" the development version of scikit-beam
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Right now you have the development version of `Scikit-beam`_, but python will not
see it.  See  :ref:`virtual_envs` for how to set up a env to work in.  Assuming you are
using conda ::

  conda create -n skbeam_dev python=3 maplotlib ipython scipy
  source activate skbeam_dev

In the directory where your copy of `Scikit-beam`_ is type::

    python setup.py develop

Several pages of output will follow the first time you do this; this wouldn't
be a bad time to get a fresh cup of coffee. At the end of it you should see
something like
``Finished processing dependencies for scikit-beam==0.0.8.post67.dev0+g1dfa736.r0``.

To make sure it has been activated **change to a different directory outside of
the scikit-beam distribution** and try this in ipython::

   In [1]: import skbeam

   In [2]: skbeam.__version__
   Out[2]: '0.0.8.post67.dev0+g1dfa736'

The actual version number will be different than in this example.  The
version string is a automatically extracted from git and follows
`PEP440`_.  For example, this version is 67 commits past the v0.0.8
tag (``'0.0.8.post67'``), the working tree is dirty (``'dev0'``) and
the last commit has the SHA ``'1dfa736'``.

.. _test_installation:

Test your development copy
^^^^^^^^^^^^^^^^^^^^^^^^^^

Testing is an important part of making sure `Scikit-beam`_ produces reliable,
reproducible results. Before you try out a new feature or think you have found
a bug make sure the tests run properly on your system.

If the test *don't* complete successfully, that is itself a bug--please
`report it <http://github.com/scikit-beam/scikit-beam/issues>`_.

To run the tests, navigate back to the directory your copy of scikit-beam is in on
your computer, then, at the shell prompt, type::

    python run_tests.py

This is another good time to get some coffee or tea. The number of test is
large. When the test are done running you will see a message something like
this::

    4741 passed, 85 skipped, 11 xfailed

Skips and xfails are fine, but if there are errors or failures please
`report them <http://github.com/scikit-beam/scikit-beam/issues>`_.

.. _try_devel:

Try out the development version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are going through this to ramp up to making more contributions to
`Scikit-beam`_ you don't actually have to do anything here.

If you are doing this because you have found a bug and are checking that it
still exists in the development version, try running your code.

Or, just for fun, try out one of the
`new features <http://scikit-beam.readthedocs.org/en/latest/changelog.html>`_ in
the development version.




.. include:: links.inc
.. _Git Basics: http://git-scm.com/book/en/Getting-Started-Git-Basics
.. _Set Up Git at GitHub: http://help.github.com/articles/set-up-git#set-up-git
.. _PEP440: https://www.python.org/dev/peps/pep-0440/
