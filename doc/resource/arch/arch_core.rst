Contents
~~~~~~~~

[[Governing Principles\|Software Architecture#governing-principles]]
guide the general architecture of the project. Governing principles will
and should be in competition with each other.

[[Architectural Decisions\|Software
Architecture#architectural-decisions]] (AD) are choices that have been
made which impact the software project in various ways. AD's can be
decisions about which programming language to use, how the codebase is
going to be structured, etc. AD's must support (at least one of) the
Governing Principles.

Governing Principles
~~~~~~~~~~~~~~~~~~~~

List of Governing Principles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Maintain provenance over time

   -  Raw data
   -  Data Analysis
   -  Process/Experimental Variables (Sample temperature, vacuum
      pressure at vacuum 1, etc.)
   -  Metadata (Principal Investigator, Run name, etc.)

-  Enable Analysis Flexibility

   -  Data pipelines
   -  Workflow management

-  Minimize the number of independent software tools required for data
   analysis

-  Enable incremental delivery of capability

-  Reuse rather than rewrite

-  Speed of delivery

-  Low entry barrier to prospective developers

   -  Assumption: Software that is easy to develop has more developers

-  Low entry barrier to prospective users

   -  Assumption: Software that is easier to use has more users

-  Enable computation on diverse hardware

   -  Commodity hardware
   -  CPU Cluster
   -  GPU Cluster

-  Design the system to be independent of User Interface (UI) delivery
   mechanism

   -  no UI (command line)
   -  Local UI
   -  Web UI

Architectural Decisions
~~~~~~~~~~~~~~~~~~~~~~~

[[Back to top\|Software Architecture#contents]]

[[Arch: Architectural Decisions Template]]

List of Architectural Decisions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`AD001`

:ref:`AD002`

:ref:`AD003`

:ref:`AD004`

:ref:`AD005`

:ref:`AD006`

:ref:`AD007`

:ref:`AD008`

:ref:`AD009`

:ref:`AD010`

:ref:`AD011`

:ref:`AD012`

:ref:`AD013`

:ref:`AD014`

:ref:`AD015`


Detailed Architectural Decisions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

[[Back to top\|Software Architecture#contents]]

.. _AD001:

AD001: Selecting Python as the Object-oriented language
'''''''''''''''''''''''''''''''''''''''''''''''''''''''
+--------------------------+------------------------------------------------+
| Property                 | Description                                    |
+==========================+================================================+
| Area of Concern          | Language Decisions                             |
+--------------------------+------------------------------------------------+
| Topic                    | nil                                            |
|                          |                                                |
+--------------------------+------------------------------------------------+
| Problem statement        | Need to establish the base and standard        |
|                          | programming language                           |
+--------------------------+------------------------------------------------+
| Assumptions              | Object-oriented language is preferred over     |
|                          | procedural                                     |
+--------------------------+------------------------------------------------+
| Motivation               | Multiple programming languages add to system   |
|                          | complexity and they slow development processes |
|                          | which result in longer delivery times          |
+--------------------------+------------------------------------------------+
| Alternatives             | Python, Java, C++                              |
+--------------------------+------------------------------------------------+
| Justification            | Python has extremely good scientific           |
|                          | support. Python requires less coding than other|
|                          | languages to accomplish the same task (Less    |
|                          | boilerplate code). Python is already a commonly|
|                          | used language in the scientific community.     |
|                          |                                                |
+--------------------------+------------------------------------------------+
| Implications             | Python is interpretative, so the CPU load is   |
|                          | higher than a compiled language. This decision |
|                          | requires that all development team members     |
|                          | become experts in Python.                      |
+--------------------------+------------------------------------------------+
| Derived Requirements     | nil                                            |
+--------------------------+------------------------------------------------+
| Related Decisions        | nil                                            |
+--------------------------+------------------------------------------------+
| Conforms to Principles   | Enable analysis flexibility, Reuse rather than |
|                          | write, Speed of delivery                       |
+--------------------------+------------------------------------------------+

.. _AD002:

AD002: Selecting Qt as the UI Toolkit
'''''''''''''''''''''''''''''''''''''
+--------------------------+----------------------------------------------------+
| Property                 | Descripton                                         |
+==========================+====================================================+
| Area of Concern          | UI Decisons                                        |
+--------------------------+----------------------------------------------------+
| Topic                    | nil                                                |
+--------------------------+----------------------------------------------------+
| Problem statement        | Need to stablish the toolkit for building the UI   |
+--------------------------+----------------------------------------------------+
| Assumptions              | nil                                                |
+--------------------------+----------------------------------------------------+
| Motivation               | nil                                                |
+--------------------------+----------------------------------------------------+
| Alternatives             | PyGTK, wPython                                     |
+--------------------------+----------------------------------------------------+
| Justification            | Qt is a opular and widely used UI toolkit. There   |
|                          | are manyvisualization tools that can be used as    |
|                          | 'drop-in, such as pyqtgraph.                       |
+--------------------------+----------------------------------------------------+
| Implications             | By usingthe Qt UI toolkit, only existing workflow  |
|                          | managemet frameworks that use Qt can be easily     |
|                          | leverage.                                          |
+--------------------------+----------------------------------------------------+
| Derived Requirements     | nil                                                |
+--------------------------+----------------------------------------------------+
| Related Decisions        | Use VisTails for UI front-end :ref:`AD014`         |
+--------------------------+----------------------------------------------------+
| Conforms to Principles   | Speed ofdelivery, Reuse rather than rewrite        |
+--------------------------+----------------------------------------------------+

.. _AD003:

AD003: Restricting base library to Python/Numpy/Scipy/Qt
''''''''''''''''''''''''''''''''''''''''''''''''''''''''
+--------------------------+-----------------------------------------------------------------------------------------+
| Property                 | Description                                                                             |
+==========================+=========================================================================================+
| Area of Concern          | Language Decisions                                                                      |
+--------------------------+-----------------------------------------------------------------------------------------+
| Topic                    | nil                                                                                     |
+--------------------------+-----------------------------------------------------------------------------------------+
| Problem statement        | nil                                                                                     |
+--------------------------+-----------------------------------------------------------------------------------------+
| Assumptions              | nil                                                                                     |
+--------------------------+-----------------------------------------------------------------------------------------+
| Motivation               | Provide base functionality with no complex install requirements.                        |
+--------------------------+-----------------------------------------------------------------------------------------+
| Alternatives             | Have all code in a single library.                                                      |
+--------------------------+-----------------------------------------------------------------------------------------+
| Justification            | Including only common libraries in our base library will allow us to leverage other     |
|                          | tools.                                                                                  |
+--------------------------+-----------------------------------------------------------------------------------------+
| Implications             | By restricting to Python/Numpy/Scipy/Qt, we might risk over-simplifying the core        |
|                          | library                                                                                 |
+--------------------------+-----------------------------------------------------------------------------------------+
| Derived Requirements     | nil                                                                                     |
+--------------------------+-----------------------------------------------------------------------------------------+
| Related Decisions        | `[AD007\|Software Architecture#ad007]                                                   |
|                          | <Analysis%20libraries%20are%20structured%20according%20to%20external%20dependencies>`__:|
|                          | Analysis libraries are structured according to external dependencies,                   |
+--------------------------+-----------------------------------------------------------------------------------------+
| Conforms to Principles   | Low entry barrier to prospective users, Low entry barrier to prospective developers,    |
|                          | Reuse rather than rewrite, Enable analysis flexibility, Enable computation on diverse   |
|                          | hardware                                                                                |
+--------------------------+-----------------------------------------------------------------------------------------+

.. _AD004:

AD004: Use Numpy docstrings for documentation
'''''''''''''''''''''''''''''''''''''''''''''
+--------------------------+----------------------------------------------------------------------------------+
| Property                 | Description                                                                      |
+==========================+==================================================================================+
| Area of Concern          |Language Decisions                                                                |
+--------------------------+----------------------------------------------------------------------------------+
| Topic                    | nil                                                                              |
+--------------------------+----------------------------------------------------------------------------------+
| Problem statement        | Need to establish the source code documentation format                           |
+--------------------------+----------------------------------------------------------------------------------+
| Assumptions              | nil                                                                              |
+--------------------------+----------------------------------------------------------------------------------+
| Motivation               | Multiple source code documentation formats do not allow for automatic            |
|                          | code-parsing tools to be used.                                                   |
+--------------------------+----------------------------------------------------------------------------------+
| Alternatives             | PEP257, PEP287                                                                   |
+--------------------------+----------------------------------------------------------------------------------+
| Justification            | Numpy documentation is widely used and supported by Sphinx.                      |
+--------------------------+----------------------------------------------------------------------------------+
| Implications             | Developers cannot be free-form with their docstrings. Numpy documentation formats|
|                          | must be adhered to.                                                              |
+--------------------------+----------------------------------------------------------------------------------+
| Derived Requirements     | nil                                                                              |
+--------------------------+----------------------------------------------------------------------------------+
| Related Decisions        | `[AD005\|Software Architecture#ad005]                                            |
|                          | <Use%20Sphinx%20for%20automatic%20generation%20of%20html%20documentation>`__: Use|
|                          | Sphinx for automatic generation of html documentation, `[AD014\|Software         |
|                          | Architecture#AD014] <Use%20VisTrails%20for%20local%20UI%20front-end>`__: Use     |
|                          | VisTrails for UI front-end                                                        |
+--------------------------+----------------------------------------------------------------------------------+
| Conforms to Principles   | Speed of delivery, Low entry barrier to prospective developers                   |
+--------------------------+----------------------------------------------------------------------------------+

.. _AD005:

AD005: Use Sphinx for automatic generation of html documentation
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
+--------------------------+-----------------------------------------------------+
| Property                 | Description                                         |
+==========================+=====================================================+
| Area of Concern          | Language Decisions                                  |
+--------------------------+-----------------------------------------------------+
| Topic                    | nil                                                 |
+--------------------------+-----------------------------------------------------+
| Problem statement        | Documentation needs to be provided in an easily     |
|                          | accessible format.                                  |
+--------------------------+-----------------------------------------------------+
| Assumptions              | The Sphinx tool will be available for the lifetime  |
|                          | of the NSLS-2 data analysis project                 |
+--------------------------+-----------------------------------------------------+
| Motivation               | Automatic generation of html documentation is a     |
|                          | project requirement.                                |
+--------------------------+-----------------------------------------------------+
| Alternatives             | PyDoc, Doxygen                                      |
+--------------------------+-----------------------------------------------------+
| Justification            | Sphinx automatically generates code documentation   |
|                          | that looks professional. Sphinx is a widely-used    |
|                          | tool, so many users will be familiar with the       |
|                          | layout and navigation of Sphinx-generated           |
|                          | documentation                                       |
+--------------------------+-----------------------------------------------------+
| Implications             | Sphinx can be challenging to set up for the novice  |
|                          | user. In order to use Sphinx, we need to keep a     |
|                          | developer on staff that is versed in the ways of    |
|                          | Sphinx                                              |
+--------------------------+-----------------------------------------------------+
| Derived Requirements     | nil                                                 |
+--------------------------+-----------------------------------------------------+
| Related Decisions        | `[AD004\|Software Architecture#ad004]               |
|                          | <Use%20Numpy%20docstrings%20for%20documentation>`__:|
|                          | Use Numpy docstrings for documentation              |
+--------------------------+-----------------------------------------------------+
| Conforms to Principles   | Low entry barrier to prospective developers, Low    |
|                          | entry barrier to prospective users, Speed of        |
|                          | delivery                                            |
+--------------------------+-----------------------------------------------------+

.. _AD006:

AD006: Store old versions of the analysis libraries in perpetuity
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
[[Back to top\|Software Architecture#contents]]

+--------------------------+------------------------------------------+
| Property                 | Description                              |
+==========================+==========================================+
| Area of Concern          | Design Decisions                         |
+--------------------------+------------------------------------------+
| Topic                    | nil                                      |
+--------------------------+------------------------------------------+
| Problem statement        | Need to maintain old versions of the     |
|                          | analysis library                         |
+--------------------------+------------------------------------------+
| Assumptions              | The git version control service will     |
|                          | continue to provide access to individual |
|                          | commits.                                 |
+--------------------------+------------------------------------------+
| Motivation               | Maintaining data analysis provenance     |
|                          | requires maintaining the exact code that |
|                          | was used to analyze the data.            |
+--------------------------+------------------------------------------+
| Alternatives             | nil                                      |
+--------------------------+------------------------------------------+
| Justification            | Without the exact code that was used to  |
|                          | generate the analysis, reputability is   |
|                          | put in jeopardy.                         |
+--------------------------+------------------------------------------+
| Implications             | Code will need to be written to allow    |
|                          | the user to select which version of the  |
|                          | analysis library they would like to run, |
|                          | if the analysis library has changed.     |
+--------------------------+------------------------------------------+
| Derived Requirements     | nil                                      |
+--------------------------+------------------------------------------+
| Related Decisions        | `[AD017\|Software Architecture#AD017]    |
|                          | <Use%20git%20for%20version%20control>`__:|
|                          | Use git for version control              |
+--------------------------+------------------------------------------+
| Conforms to Principles   | Maintain provenance over time            |
+--------------------------+------------------------------------------+

.. _AD007:

AD007: Analysis libraries are structured according to external dependencies
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
+--------------------------+----------------------------------------------------------------+
| Property                 | Description                                                    |
+==========================+================================================================+
| Area of Concern          | Design Decisions                                               |
+--------------------------+----------------------------------------------------------------+
| Topic                    | nil                                                            |
+--------------------------+----------------------------------------------------------------+
| Problem statement        | How is the codebase going to be structured?                    |
+--------------------------+----------------------------------------------------------------+
| Assumptions              | Complex external dependencies are bad                          |
+--------------------------+----------------------------------------------------------------+
| Motivation               | There are too many possible external dependencies for data     |
|                          | analysis to contain all codes in a single library              |
+--------------------------+----------------------------------------------------------------+
| Alternatives             | Put all code and dependencies in a single library              |
+--------------------------+----------------------------------------------------------------+
| Justification            | By separating code into libraries based on external            |
|                          | dependencies, managing the codebase becomes easier. Separating |
|                          | code by external dependencies allows for modular installation  |
|                          | where only the tools that the user wants are installed. This   |
|                          | significantly simplifies installation.                         |
+--------------------------+----------------------------------------------------------------+
| Implications             | Managing multiple libraries is more complex than managing one  |
|                          | since each library has significant boilerplate: documentation, |
|                          | install scripts, folder structure, etc.                        |
+--------------------------+----------------------------------------------------------------+
| Derived Requirements     | nil                                                            |
+--------------------------+----------------------------------------------------------------+
| Related Decisions        | `[AD003\|Software Architecture#ad003]                          |
|                          | <Restricting%20base%20library%20to%20Python/Numpy/Scipy/Qt>`__:|
|                          | Restricting base library to Python/Numpy/Scipy/Qt              |
+--------------------------+----------------------------------------------------------------+
| Conforms to Principles   | Enable computation on diverse hardware                         |
+--------------------------+----------------------------------------------------------------+

.. _AD008:

AD008: Data types are standardized by library
''''''''''''''''''''''''''''''''''''''''''''
+--------------------------+-----------------------------------------------------------------------------------------+
| Property                 | Description                                                                             |
+==========================+=========================================================================================+
| Area of Concern          | Design Decisions                                                                        |
+--------------------------+-----------------------------------------------------------------------------------------+
| Topic                    | nil                                                                                     |
+--------------------------+-----------------------------------------------------------------------------------------+
| Problem statement        | Data types need to be standardized                                                      |
+--------------------------+-----------------------------------------------------------------------------------------+
| Assumptions              | Non-standardized data types are bad                                                     |
+--------------------------+-----------------------------------------------------------------------------------------+
| Motivation               | The data types a library can use must be clearly defined                                |
+--------------------------+-----------------------------------------------------------------------------------------+
| Alternatives             | nil                                                                                     |
+--------------------------+-----------------------------------------------------------------------------------------+
| Justification            | Clearly defined data types make a library easier to understand                          |
+--------------------------+-----------------------------------------------------------------------------------------+
| Implications             | The possible inputs and outputs to data analysis functions become more constrained.     |
+--------------------------+-----------------------------------------------------------------------------------------+
| Derived Requirements     | nil                                                                                     |
+--------------------------+-----------------------------------------------------------------------------------------+
| Related Decisions        | `[AD007\|Software Architecture#ad007]                                                   |
|                          | <Analysis%20libraries%20are%20structured%20according%20to%20external%20dependencies>`__:|
|                          | Analysis libraries are structured according to external dependencies                    |
+--------------------------+-----------------------------------------------------------------------------------------+
| Conforms to Principles   | Low entry barrier to prospective developers, Reuse rather than rewrite, Enable          |
|                          | computation on diverse hardware                                                         |
+--------------------------+-----------------------------------------------------------------------------------------+

.. _AD009:

AD009: All local UI tools are created as 'qt' widgets
''''''''''''''''''''''''''''''''''''''''''''''''''''
+--------------------------+----------------------------------------------------------+
| Property                 | Description                                              |
+==========================+==========================================================+
| Area of Concern          | UI Decisions                                             |
+--------------------------+----------------------------------------------------------+
| Topic                    | nil                                                      |
+--------------------------+----------------------------------------------------------+
| Problem statement        | UI tool development needs to follow a standardized       |
|                          | procedure to enforce maximum flexibility.                |
+--------------------------+----------------------------------------------------------+
| Assumptions              | nil                                                      |
+--------------------------+----------------------------------------------------------+
| Motivation               | Standardizing UI tool development                        |
+--------------------------+----------------------------------------------------------+
| Alternatives             | Allow UI developers to construct tools however they see  |
|                          | fit                                                      |
+--------------------------+----------------------------------------------------------+
| Justification            | By forcing every UI tool to subclass 'QtGui.QWidget' they|
|                          | become modular. Additionally, they can be easily plugged |
|                          | in to any front-end UI that uses Qt (e.g., frameworks    |
|                          | like VisTrails)                                          |
+--------------------------+----------------------------------------------------------+
| Implications             | All widgets require additional code to make them         |
|                          | stand-alone tools.                                       |
+--------------------------+----------------------------------------------------------+
| Derived Requirements     | nil                                                      |
+--------------------------+----------------------------------------------------------+
| Related Decisions        | nil                                                      |
+--------------------------+----------------------------------------------------------+
| Conforms to Principles   | Enable analysis flexibility                              |
+--------------------------+----------------------------------------------------------+

.. _AD010:

AD010: All UI widgets are built as independent, composable elements
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
+--------------------------+----------------------+
| Property                 | Description          |
+==========================+======================+
| Area of Concern          | Language Decisions   |
+--------------------------+----------------------+
| Topic                    | nil                  |
+--------------------------+----------------------+
| Problem statement        | nil                  |
+--------------------------+----------------------+
| Assumptions              | nil                  |
+--------------------------+----------------------+
| Motivation               | nil                  |
+--------------------------+----------------------+
| Alternatives             | nil                  |
+--------------------------+----------------------+
| Justification            | nil                  |
+--------------------------+----------------------+
| Implications             | nil                  |
+--------------------------+----------------------+
| Derived Requirements     | nil                  |
+--------------------------+----------------------+
| Related Decisions        | nil                  |
+--------------------------+----------------------+
| Conforms to Principles   | nil                  |
+--------------------------+----------------------+


.. _AD011:

AD011: Input is modular
''''''''''''''''''''''
[[Back to top\|Software Architecture#contents]]

+--------------------------+--------------------------------------------------------------+
| Property                 | Description                                                  |
+==========================+==============================================================+
| Area of Concern          | Design Decisions                                             |
|                          |                                                              |
+--------------------------+--------------------------------------------------------------+
| Topic                    | nil                                                          |
+--------------------------+--------------------------------------------------------------+
| Problem statement        | Data needs to be input from multiple formats                 |
|                          |                                                              |
|                          |                                                              |
|                          |                                                              |
+--------------------------+--------------------------------------------------------------+
| Assumptions              | Input data formats cannot be reduced to a single type        |
|                          |                                                              |
|                          |                                                              |
|                          |                                                              |
|                          |                                                              |
+--------------------------+--------------------------------------------------------------+
| Motivation               | Support data formats from any source.                        |
|                          |                                                              |
|                          |                                                              |
+--------------------------+--------------------------------------------------------------+
| Alternatives             | Only support input data from the Data Broker. Only support   |
|                          |input data from a hard-coded list of formats.                 |
|                          |                                                              |
|                          |                                                              |
|                          |                                                              |
|                          |                                                              |
|                          |                                                              |
|                          |                                                              |
|                          |                                                              |
+--------------------------+--------------------------------------------------------------+
| Justification            | Data comes in many flavors from many sources. By restricting |
|                          | our software to use only one kind of input data is to        |
|                          | automatically reduce the user base.                          |
|                          |                                                              |
|                          |                                                              |
+--------------------------+--------------------------------------------------------------+
| Implications             | Allowing input data to be modular requires that the software |
|                          | be designed more carefully to allow such flexibility.        |
|                          |                                                              |
+--------------------------+--------------------------------------------------------------+
| Derived Requirements     | nil                                                          |
+--------------------------+--------------------------------------------------------------+
| Related Decisions        | nil                                                          |
+--------------------------+--------------------------------------------------------------+
| Conforms to Principles   | Enable analysis flexibility                                  |
|                          |                                                              |
|                          |                                                              |
+--------------------------+--------------------------------------------------------------+

.. _AD012:

AD012: Output is modular
''''''''''''''''''''''''
+--------------------------+--------------------------------------------------------------+
| Property                 | Description                                                  |
+==========================+==============================================================+
| Area of Concern          | Design Decisions                                             |
+--------------------------+--------------------------------------------------------------+
| Topic                    | nil                                                          |
+--------------------------+--------------------------------------------------------------+
| Problem statement        | Data needs to be output in multiple formats                  |
+--------------------------+--------------------------------------------------------------+
| Assumptions              | Output data formats cannot be reduced to a single type       |
+--------------------------+--------------------------------------------------------------+
| Motivation               | Support data formats from any source.                        |
+--------------------------+--------------------------------------------------------------+
| Alternatives             | Only support output data to the Data Broker. Only support    |
|                          | output data to a hard-coded list of formats.                 |
+--------------------------+--------------------------------------------------------------+
| Justification            | Data comes in many flavors from many sources. By restricting |
|                          | our software to only output to a single data format is to    |
|                          | automatically reduce the user base. Consider Reitveld        |
|                          | refinement; a critical component of X-ray Powder Diffraction |
|                          | (XPD). There are multiple popular packages to perform        |
|                          | Reitveld refinement, each of which has their own data type,   |
|                          | and each of which our software needs to support.             |
|                          |                                                              |
+--------------------------+--------------------------------------------------------------+
| Implications             | Allowing output data to be modular requires that the software|
|                          |be designed more carefully to allow such flexibility.         |
|                          |                                                              |
+--------------------------+--------------------------------------------------------------+
| Derived Requirements     | nil                                                          |
+--------------------------+--------------------------------------------------------------+
| Related Decisions        | nil                                                          |
+--------------------------+--------------------------------------------------------------+
| Conforms to Principles   | Enable analysis flexibility                                  |
+--------------------------+--------------------------------------------------------------+

.. _AD013:

AD013: Local UI tools do no calculation and only provide an interface for
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


+--------------------------+--------------------+
| Property                 | Description        |
+==========================+====================+
| Area of Concern          | Design Decisions   |
+--------------------------+--------------------+
| Topic                    | nil                |
+--------------------------+--------------------+
| Problem statement        | nil                |
+--------------------------+--------------------+
| Assumptions              | nil                |
+--------------------------+--------------------+
| Motivation               | nil                |
+--------------------------+--------------------+
| Alternatives             | nil                |
+--------------------------+--------------------+
| Justification            | nil                |
+--------------------------+--------------------+
| Implications             | nil                |
+--------------------------+--------------------+
| Derived Requirements     | nil                |
+--------------------------+--------------------+
| Related Decisions        | nil                |
+--------------------------+--------------------+
| Conforms to Principles   | nil                |
+--------------------------+--------------------+

.. _AD014:

AD014: Use VisTrails for the local UI front-end
'''''''''''''''''''''''''''''''''''''''''''''''

+--------------------------+----------------------------------------------------------+
| Property                 | Description                                              |
+==========================+==========================================================+
| Area of Concern          | UI Decisions                                             |
+--------------------------+----------------------------------------------------------+
| Topic                    | nil                                                      |
+--------------------------+----------------------------------------------------------+
| Problem statement        | A workflow manager is a project requirement              |
+--------------------------+----------------------------------------------------------+
| Assumptions              | This data analysis project requires a workflow manager   |
+--------------------------+----------------------------------------------------------+
| Motivation               | nil                                                      |
+--------------------------+----------------------------------------------------------+
| Alternatives             | Mantid, DAWN                                             |
+--------------------------+----------------------------------------------------------+
| Justification            | nil                                                      |
+--------------------------+----------------------------------------------------------+
| Implications             | nil                                                      |
+--------------------------+----------------------------------------------------------+
| Derived Requirements     | nil                                                      |
+--------------------------+----------------------------------------------------------+
| Related Decisions        | nil                                                      |
+--------------------------+----------------------------------------------------------+
| Conforms to Principles   | nil                                                      |
+--------------------------+----------------------------------------------------------+

.. _AD015:

AD015: Plugins for Control Systems Studio will not be developed
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
+--------------------------+-----------------------------------------------------------------+
| Property                 | Description                                                     |
+==========================+=================================================================+
| Area of Concern          | UI Decisions                                                    |
+--------------------------+-----------------------------------------------------------------+
| Topic                    | nil                                                             |
+--------------------------+-----------------------------------------------------------------+
| Problem statement        | Plugins for Control Systems Studio (CSS) are too complex to     |
|                          | write                                                           |
+--------------------------+-----------------------------------------------------------------+
| Assumptions              | Plugins for CSS will always be more challenging to write than   |
|                          | Python-based QT widgets                                         |
+--------------------------+-----------------------------------------------------------------+
| Motivation               | Because none of the developers on this data analysis project    |
|                          | have experience with CSS, we will not be developing plugins for |
|                          | the Eclipse-based tool.                                         |
+--------------------------+-----------------------------------------------------------------+
| Alternatives             | Write plugins for CSS.                                          |
+--------------------------+-----------------------------------------------------------------+
| Justification            | In addition to not having developers that have experience in    |
|                          | developing plugins for the Eclipse platform, it seems to be     |
|                          | commonly understood that developing Eclipse plugins is an       |
|                          | arduous and challenging task.                                   |
+--------------------------+-----------------------------------------------------------------+
| Implications             | By not writing plugins for CSS, we are requiring users to       |
|                          | control the beamline with one tool and perform most of their    |
|                          | data analysis with a separate tool.                             |
+--------------------------+-----------------------------------------------------------------+
| Derived Requirements     | nil                                                             |
+--------------------------+-----------------------------------------------------------------+
| Related Decisions        | `[AD001\|Software Architecture#ad001]                           |
|                          | <Selecting%20Python%20as%20the%20Object-oriented%20language>`__:|
|                          | Selecting Python as the Object-oriented language,\              |
|                          | `[AD002\|Software Architecture#ad002]                           |
|                          | <Selecting%20Qt%20as%20the%20local%20UI%20Toolkit>`__: Selecting|
|                          | Qt as the UI Toolkit,[[AD009\|Software Architecture#ad009]]: All|
|                          | UI tools are created as 'qt' widgets,[[AD010\|Software          |
|                          | Architecture#ad010]]: All UI widgets are built as independent,  |
|                          | composable elements,\ `[AD014\|Software Architecture#AD014]     |
|                          | <Use%20VisTrails%20for%20local%20UI%20front-end>`__: Use        |
|                          | VisTrails for UI front-end                                      |
+--------------------------+-----------------------------------------------------------------+
| Conforms to Principles   | Speed of delivery, Low entry barrier to prospective developers  |
+--------------------------+-----------------------------------------------------------------+

.. _AD016:

AD016: Coding style will follow Python PEP8
''''''''''''''''''''''''''''''''''''''''''
[[Back to top\|Software Architecture#contents]]

+--------------------------+----------------------------------------------------------------+
| Property                 | Description                                                    |
+==========================+================================================================+
| Area of Concern          | Language Decisions                                             |
+--------------------------+----------------------------------------------------------------+
| Topic                    | nil                                                            |
+--------------------------+----------------------------------------------------------------+
| Problem statement        | Coding standards are useful                                    |
+--------------------------+----------------------------------------------------------------+
| Assumptions              | Coders require coding standards.                               |
+--------------------------+----------------------------------------------------------------+
| Motivation               | Without coding standards, chaos will reign                     |
+--------------------------+----------------------------------------------------------------+
| Alternatives             | chaos                                                          |
+--------------------------+----------------------------------------------------------------+
| Justification            | PEP8 is a community-defined coding standard that is widely-used|
|                          | and accepted                                                   |
+--------------------------+----------------------------------------------------------------+
| Implications             | Coders will need to learn and follow PEP8 coding standards. Or |
|                          | instruct their IDE to follow the PEP8 standard.                |
+--------------------------+----------------------------------------------------------------+
| Derived Requirements     | nil                                                            |
+--------------------------+----------------------------------------------------------------+
| Related Decisions        | nil                                                            |
+--------------------------+----------------------------------------------------------------+
| Conforms to Principles   | Low entry barrier to prospective developers                    |
+--------------------------+----------------------------------------------------------------+

.. _AD017:

AD017: Use git for version control
'''''''''''''''''''''''''''''''''
+--------------------------+---------------------------------------------------------------------------------+
| Property                 | Description                                                                     |
+==========================+=================================================================================+
| Area of Concern          | Language Decisions                                                              |
+--------------------------+---------------------------------------------------------------------------------+
| Topic                    | nil                                                                             |
+--------------------------+---------------------------------------------------------------------------------+
| Problem statement        | All coding projects should use version control                                  |
+--------------------------+---------------------------------------------------------------------------------+
| Assumptions              | Version control is always useful                                                |
+--------------------------+---------------------------------------------------------------------------------+
| Motivation               | Version control allows for distributed development, among many other benefits   |
+--------------------------+---------------------------------------------------------------------------------+
| Alternatives             | none, really                                                                    |
+--------------------------+---------------------------------------------------------------------------------+
| Justification            | Version control has many benefits. Google "reasons to use version control"      |
+--------------------------+---------------------------------------------------------------------------------+
| Implications             | Developers will need to learn how to use git                                    |
+--------------------------+---------------------------------------------------------------------------------+
| Derived Requirements     | nil                                                                             |
+--------------------------+---------------------------------------------------------------------------------+
| Related Decisions        | `[AD006\|Software Architecture#ad006]                                           |
|                          | <Store%20old%20versions%20of%20the%20analysis%20libraries%20in%20perpetuity>`__:|
|                          | Store old versions of the analysis libraries in perpetuity                      |
+--------------------------+---------------------------------------------------------------------------------+
| Conforms to Principles   | Maintain provenance over time, Enable incremental delivery of capability, Speed |
|                          | of delivery, Low entry barrier to prospective developers                        |
+--------------------------+---------------------------------------------------------------------------------+

.. _AD018:

AD018: Variable names are standardized across all NSLS2 libraries
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
+--------------------------+----------------------+
| Property                 | Description          |
+==========================+======================+
| Area of Concern          | Language Decisions   |
+--------------------------+----------------------+
| Topic                    | nil                  |
+--------------------------+----------------------+
| Problem statement        | nil                  |
+--------------------------+----------------------+
| Assumptions              | nil                  |
+--------------------------+----------------------+
| Motivation               | nil                  |
+--------------------------+----------------------+
| Alternatives             | nil                  |
+--------------------------+----------------------+
| Justification            | nil                  |
+--------------------------+----------------------+
| Implications             | nil                  |
+--------------------------+----------------------+
| Derived Requirements     | nil                  |
+--------------------------+----------------------+
| Related Decisions        | nil                  |
+--------------------------+----------------------+
| Conforms to Principles   | nil                  |
+--------------------------+----------------------+

.. _AD021:

AD021: Explicitly declare and isolate dependencies
''''''''''''''''''''''''''''''''''''''''''''''''''
+--------------------------+----------------------------------------------------------+
| Property                 | Description                                              |
+==========================+==========================================================+
| Area of Concern          | Design Decisions                                         |
+--------------------------+----------------------------------------------------------+
| Topic                    | nil                                                      |
+--------------------------+----------------------------------------------------------+
| Problem statement        | nil                                                      |
+--------------------------+----------------------------------------------------------+
| Assumptions              | Complex dependencies are prohibitive to prospective users|
|                          | and developers                                           |
+--------------------------+----------------------------------------------------------+
| Motivation               | nil                                                      |
+--------------------------+----------------------------------------------------------+
| Alternatives             | N/A                                                      |
+--------------------------+----------------------------------------------------------+
| Justification            | Explicit dependency declaration simplifies setup for new |
|                          | developers. While certain tools exist on many systems,   |
|                          | there is no guarantee that they will exist on *all*      |
|                          | systems where our library may be used now or in the      |
|                          | future, reducing the functionality and portability of our|
|                          | codebase                                                 |
+--------------------------+----------------------------------------------------------+
| Implications             | Any code that depends on the existence of system-wide    |
|                          | tools will not be functional on a system that does not   |
|                          | have that tool. Installing the missing tools is not      |
|                          | always a trivial task and will increase the entry barrier|
|                          | to prospective developers.                               |
+--------------------------+----------------------------------------------------------+
| Derived Requirements     | nil                                                      |
+--------------------------+----------------------------------------------------------+
| Related Decisions        | nil                                                      |
+--------------------------+----------------------------------------------------------+
| Conforms to Principles   | nil                                                      |
+--------------------------+----------------------------------------------------------+
