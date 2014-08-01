.. _logging:

Logging
=======

Getting feed back from running programs is invaluable for assessing
the health and performance of the code.  However copious ``print``
statements are not practical on projects larger than short scripts.
This is particularly true for libraries which are imported into user
code; it is rude to spam the user with debugging output.

This is such a common need that the tools to solve it are built into
the core python library in the :mod:`logging` module.

 - `Demonstration <https://github.com/tacaswell/logger_demo>`_
 - `Basic tutorial <https://docs.python.org/2/howto/logging.html>`_
 - `Detailed reference <https://docs.python.org/2/library/logging.html>`_
 - `Cookbook <https://docs.python.org/2/howto/logging-cookbook.html>`_

Rough Overview
--------------

The logging module provides a frame work for generating and
propagating messages.  Each process has a hierarchy of :class:`Logger`
objects.  Using the methods on these objects you can generate error
messages with a severity level attached.  The log messages are then
formatted (using a :class:`Formatter` object) and distributed by
:class:`Handler` objects attached to the :class:`Logger`.  The
messages are also passed up to any parent :class:`Logger` s.  Each
:class:`Handler` and :class:`Logger` objects have a severity threshold, messages
below that threshold are ignored.  This enables easy run-time selection of the
verbosity of the logging.


There are five default levels of logging, listed in decreasing order of
severity:

+-------------------------+-----------------------------------------------+
|Level                    |Description                                    |
|                         |                                               |
+=========================+===============================================+
|Critical                 |The program may crash in the near future,      |
|                         |things have gone very sideways.                |
|                         |                                               |
+-------------------------+-----------------------------------------------+
|Error/Exception          |Something has gone badly wrong, an operation   |
|                         |failed                                         |
|                         |                                               |
+-------------------------+-----------------------------------------------+
|Warning                  |Something has gone slightly wrong or might go  |
|                         |wrong in the future                            |
|                         |                                               |
+-------------------------+-----------------------------------------------+
|Info                     |Status, indications everything is working      |
|                         |correctly.                                     |
|                         |                                               |
+-------------------------+-----------------------------------------------+
|Debug                    |Messages that are useful for debugging, but    |
|                         |are too detailed to be generally useful        |
|                         |                                               |
+-------------------------+-----------------------------------------------+

Nuts and Bolts
--------------
The loggers are hierarchical (by dotted name).  If a logger does not have
a level explicitly set, it will use the level of it's parent.  Unless prohibited
loggers will forward all of their accepted messages to their parents.

Create a message
````````````````
A :code:`logger` is defined in each module of our libraries by :code:`logger =
logging.getLogger(__name__)` where :code:`__name__` is the module name.
Creating messages with the various severity levels is done by ::

    logger.debug("this is a debug message")
    logger.info("this is a info message")
    logger.warning("this is a warning message")
    logger.error("this is a error message")
    logger.critical("this is a critical message")

which will yield an error message with the body "this is a [level] message".

The error messages also understand basic string formatting so ::

    logger.debug("this is a %s debug message no. %d", great, 42)

will yield a message with the body "this is a great debug message no. 42".

Attaching a Handler
```````````````````

By default the library does not attach a non-null :class:`Handler` to
any of the :class:`Logger` objects (`see
<https://docs.python.org/2/howto/logging.html#configuring-logging-for-a-library>`_).
In order to get the messages out a :class:`Handler` (with it's own
:class:`Formatter`) need to be attached to the logger ::

    h = logging.StreamHandler()
    form = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger.addHandler(h)

The above code demonstrates the mechanism by which a `StreamHandler` is
attached to the logger.  `StreamHandler` writes to stderr by default.


`Detailed explanations of the available handlers <https://docs.python.org/2/howto/logging.html#useful-handlers>`_.




Defining a Formatter
````````````````````

The :class:`Formatters` are essentially string formatting.  For a full
list of the data available and the corresponding variable names, see `this list
<https://docs.python.org/2/library/logging.html#logrecord-attributes>`_

For example to show the time, the severity, and the message ::

    form = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

Or to see the time, the level as a number, the function the logging call was in and
the message ::

    form = logging.Formatter('%(asctime)s - %(levelno)s - %(funcName)s - %(message)s')

Or to completely dis-regard everything ::

    form = logging.Formatter('HI MOM')
