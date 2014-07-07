.. _workflow:

Git workflow
^^^^^^^^^^^^

The basic workflow is that you develop new code on feature branches in
your local repository.  When you want your code merged into the main
repository you push your work to ``github`` and create a "Pull
Request" (PR).  The code is then reviewed and once everyone is happy,
it will be merged into ``NLSL-II/master``.

Rules
-----

  1. Don't work directly on the main repository, push commits to your
     personal github repository
  2. Never work on ``master``, always work on a feature branch
  3. Don't merge your own PR.
  4. Don't merge ``NSLS-II/master`` into your feature branch
     as this can make merging back into master tricky.
  5. Commit messages must be at least one sentence and preferably a short
     paragraph.  They should not be longer than 2 paragraphs (if you need to
     write that much the commit should probably be split up).
  6. Commit messages should start with a descriptor code (borrowed from numpy
     ), see table below.
  7. Add a file in ``changelog`` describing the propose of this branch
     in a sentence.  When we make releases these will be compiled into
     a ``CHANGELOG``.  Doing it this way prevents excessive merge conflicts.
  8. Add a test for every new function you add
  9. When you fix a bug add a test that used to fail and now passes

====  ===
Code  description
====  ===
API   an (incompatible) API change
BLD   change related to building
BUG   bug fix
DEP   deprecate something, or remove a deprecated object
DEV   development tool or utility
DOC   documentation
ENH   enhancement
MNT   maintenance commit (refactoring, typos, etc.)
REV   revert an earlier commit
STY   style fix (whitespace, PEP8)
TST   addition or modification of tests
REL   related to releasing numpy
WIP   Commit that is a work in progress
====  ===


We should try to conform to pep8.

Example
-------

Assuming you already have your repository set up::

   git fetch upstream
   git checkout master  # switch to master
   git merge upstream/master  # make sure master is up-to-date
   git checkout -b new_feature

You now do a bunch of work::

   git add ...
   git commit
   git add ...
   git commit

and when you are happy with it **push** it to github ::

   git push --set-upstream github new_feature

On the github webpage navigate to your repository and there should be a
green button that says 'Compare and Pull request'.  Push this button and
fill out the form.  This will create the PR and the change can be discussed
on github.  To make changes to the PR all you need to do is **push** more
commits to that branch on github ::

   git add ...
   git commit
   git push github

Once everyone is happy with the changes, the branch can be merged into
the main ``master`` branch via the web interface.  If there are
conflicts on ``master`` with your work, it is your responsibility to
**rebase** your branch on to the current ``master``.
