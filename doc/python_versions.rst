Supported versions of Python
----------------------------

The primary development target of the ``scikit-beam`` is python 3.5+.
Affiliated packages are encouraged, but not rquired to support legacy
python when practical.  The core library will support 2.7 as long as
the upstream scientific libraries do.

Upstream (CPython) has made in very clear that python2 will not be
supported going forward and that `no new features will be added to
python 2 <https://www.python.org/dev/peps/pep-0404/>`__ .  The EOL for
python 2.7 was already extended from `2015 to 2020
<http://legacy.python.org/dev/peps/pep-0373/>`__ to fix `critical
network security issues <https://www.python.org/dev/peps/pep-0466/>`__
for entities that have large network facing code bases they can not
quickly migrate.

All of the core libraries of the scientific stack fully support python
3.x and support for python 2.6 has been dropped for matplotlib and
pandas and is scheduled to be dropped by numpy.  The discussion in the
community is not 'if' to drop legacy python support, but 'when' and
'how'.

Moving to python 3.5+ will greatly simplify supporting c-extensions on
windows.  Currently, to compile c-extensions from legacy python
requires using an unsupported version of visual studio (which was
re-released as a 'community edition' for the sole reason of supporting
python).  Due to changes in the MS c runtime c-extensions will always
be able to be compiled with the current version of visual studio.
